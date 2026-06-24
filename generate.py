import json
import logging
import random
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import albumentations as alb
import augraphy as aug
import cv2
import matplotlib.pyplot as plt
import music21 as m21
import numpy as np
from matplotlib import patches
from PIL import Image
from tqdm.auto import tqdm

import image_ops
import score_ops
from proc import Inkscape, MuseScore, MXMLProcessor, SVGProcessor, Verovio


def showimg(img):
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    plt.close()


def plot_bboxes(image, boxes, output: Path | None = None):
    fig = plt.figure()
    ax = plt.axes()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cmap = plt.get_cmap("tab20")  # good discrete palette

    for ii, (x, y, w, h) in enumerate(boxes):
        color = cmap(ii % cmap.N)
        ax.add_patch(
            patches.Rectangle(
                xy=(x, y),
                width=w,
                height=h,
                fill=False,  # outlined only
                edgecolor=color,  # increment color per box
                linewidth=1.0,
            )
        )
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
    plt.close(fig)


BACKGROUND_PATH = Path("./backgrounds")

INK_AUGMENTATIONS = [
    aug.InkBleed(
        intensity_range=(0.4, 0.9),
        kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
        severity=(0.2, 0.9),
        p=0.3,
    ),
    aug.BleedThrough(
        intensity_range=(0.2, 0.8),
        color_range=(32, 224),
        ksize=(15, 31),
        sigmaX=1,
        alpha=random.uniform(0.1, 0.4),
        offsets=(10, 20),
        p=0.25,
    ),
    aug.InkShifter(
        text_shift_scale_range=(18, 36),
        text_shift_factor_range=(1, 5),
        text_fade_range=(0, 2),
        blur_kernel_size=(11, 11),
        blur_sigma=0.1,
        noise_type="random",
        p=0.25,
    ),
    aug.Letterpress(
        n_samples=(300, 800),
        n_clusters=(300, 800),
        std_range=(1500, 5000),
        value_range=(100, 220),
        value_threshold_range=(100, 255),
        blur=1,
        p=0.25,
    ),
    aug.InkColorSwap(
        ink_swap_color="random",
        ink_swap_sequence_number_range=(5, 10),
        ink_swap_min_width_range=(2, 3),
        ink_swap_max_width_range=(100, 120),
        ink_swap_min_height_range=(2, 3),
        ink_swap_max_height_range=(100, 120),
        ink_swap_min_area_range=(10, 40),
        ink_swap_max_area_range=(400, 500),
        p=0.5,
    ),
    aug.LinesDegradation(),
]

PAPER_AUGMENTATIONS = [
    image_ops.PaperFactory(
        texture_path=str(BACKGROUND_PATH),
        texture_enable_color=1,
        texture_color="Old",
        p=1.0,
    ),
    aug.ColorShift(),
]

bbox_params = alb.BboxParams(coord_format="coco")
ALBUMENTATION_PIPELINE = alb.Compose(
    [
        alb.ElasticTransform(
            alpha=4.0,
            p=1,
            fill=(255, 255, 255),
        ),
        alb.Affine(
            scale=(1.0, 1.2),
            shear=(-3, 3),
            p=1.0,
            fill=(255, 255, 255),
            fit_output=True,
        ),
        alb.SaltAndPepper(amount_range=(0.01, 0.02), salt_vs_pepper_range=(1.0, 0.0)),
    ],
    bbox_params=bbox_params,
)


def generate_musicxmls(source_path: Path, target_path: Path) -> None:
    for ii, source_path in enumerate(source_path.rglob("*.mxl")):
        try:
            score = m21.converter.parse(source_path)
        except Exception:
            logging.warning(f"Skipped {source_path} because it cannot be parsed")
            continue
        if not isinstance(score, m21.stream.Score):
            logging.warning(
                f"Skipping score because {source_path} does not load into the correct stream"
            )
            continue
        score.metadata = None
        score = score_ops.turn_score_monophonic(score)
        for jj, part in enumerate(score.parts):
            for measure_start in range(
                len(part.recurse().getElementsByClass(m21.stream.Measure)) - 7
            ):
                for score_size in range(4, 8):
                    measure_end = measure_start + score_size
                    score_slice = part.measures(
                        measure_start, measure_end, indicesNotNumbers=True
                    )
                    if (
                        len(score_slice.recurse().getElementsByClass(m21.note.Note)) < 6
                        or len(score_slice.recurse().getElementsByClass(m21.note.Note))
                        > 128
                    ):
                        logging.info(
                            f"Skipping S{ii} P{jj} {measure_start}: {measure_end} because it has < 6 notes or > 128"
                        )
                        continue

                    for fifths in range(-6, 6):
                        variation = fifths + 7
                        transposed = score_slice.transpose(fifths)
                        if transposed is None:
                            continue
                        score_ops.randomly_convert_to_rest(transposed)
                        score_ops.randomly_modify_pitches(transposed)
                        transposed = score_ops.rebuild_beams(transposed)
                        score_path = (
                            target_path
                            / f"w{ii}_p{jj}_m{measure_start}to{measure_end}_v{variation}.musicxml"
                        )
                        try:
                            transposed.write(
                                "musicxml",
                                score_path,
                            )
                        except Exception as exc:
                            logging.warning(
                                "Exception while exporting musicxml: {exc}. Skipping..."
                            )


def render(
    source_path: Path,
    target_path: Path,
    create_debug_images: bool = False,
) -> None:
    for musicxml_path in tqdm(
        list(source_path.glob("*.musicxml")), desc="Rendering scores..."
    ):
        try:
            mxml_proc = MXMLProcessor()
            svg_proc = SVGProcessor()

            mxml_proc.process(musicxml_path)
            svg_path = musicxml_path.with_suffix(".svg")
            png_path = musicxml_path.with_suffix(".png")
            json_path = musicxml_path.with_suffix(".json")
            Verovio.run(musicxml_path, svg_path)
            svg_proc.process(svg_path)
            id2smufl = {
                mxml_id: smufl_id
                for _, mxml_id, smufl_id in svg_proc.extract_objects(svg_path)
            }

            Inkscape.run(svg_path, png_path)
            bboxes = {
                ident: {**values, "smufl_id": id2smufl[ident]}
                for ident, values in Inkscape.run_bboxes(svg_path).items()
                if ident in id2smufl
            }
            bboxes_albumentations = np.array(
                [
                    [
                        params["x"] - 1,
                        params["y"] - 1,
                        params["w"] + 2,
                        params["h"] + 2,
                    ]
                    for _, params in bboxes.items()
                ]
            ).astype(int)
            metadata = [
                (mxml_id, params["smufl_id"]) for mxml_id, params in bboxes.items()
            ]
            with open(json_path, "w") as f_json:
                json.dump(bboxes, f_json, indent=4)

            pipeline = aug.AugraphyPipeline(
                INK_AUGMENTATIONS,
                PAPER_AUGMENTATIONS,
                [],
                random_seed=18,
            )
            foreground = np.array(Image.open(png_path).convert("RGB"))
            foreground[foreground > 64] = 255
            albumentation_output = ALBUMENTATION_PIPELINE(
                image=foreground,
                bboxes=bboxes_albumentations,
            )
            augraphy_output = pipeline(albumentation_output["image"])
            if augraphy_output is None:
                raise ValueError("Augraphy did not produce an image")

            post_bboxes = {
                mxml_id: {"smufl_id": smufl_id, "x": x, "y": y, "w": w, "h": h}
                for (x, y, w, h), (mxml_id, smufl_id) in zip(
                    albumentation_output["bboxes"].astype(int).tolist(), metadata
                )
            }

            new_musicxml_path = target_path / musicxml_path.name
            output_png_path = target_path / png_path.name
            output_bbox_path = target_path / f"{png_path.stem}.json"

            if create_debug_images:
                output_png_debug_path = output_png_path.with_stem(
                    f"{output_png_path.stem}_debug"
                )
                plot_bboxes(
                    augraphy_output,
                    [(x["x"], x["y"], x["w"], x["h"]) for x in post_bboxes.values()],
                    output_png_debug_path,
                )

            shutil.copy(musicxml_path, new_musicxml_path)
            augrahpy_output = cv2.cvtColor(augraphy_output, cv2.COLOR_BGR2RGB)
            Image.fromarray(augrahpy_output).save(output_png_path)

            with open(output_bbox_path, "w") as f_json:
                json.dump(post_bboxes, f_json, indent=4)
        except Exception as exc:
            logging.warning(f"Found error while processing {musicxml_path}: {exc}")
            raise


def setup() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "musicxml_path",
        type=Path,
        help="Path to a collection of MusicXML files (in any nesting)",
    )
    parser.add_argument(
        "clean_path",
        type=Path,
        help="Path to store the clean version of the images.",
    )
    parser.add_argument(
        "augmented_path",
        type=Path,
        help="Path to store the augmented version of the images.",
    )
    parser.add_argument(
        "--logging_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--omit_musicxml_generation",
        action="store_true",
        help="Do not generate MusicXML files again",
    )
    parser.add_argument(
        "--verovio_path",
        type=Path,
        default=Path("verovio"),
        help="Path to the verovio executable",
    )
    parser.add_argument(
        "--inkscape_path",
        type=Path,
        default=Path("inkscape"),
        help="Path to the inkscape executable",
    )
    parser.add_argument(
        "--create_debug_images",
        action="store_true",
        help="Create debug images with overlaid bounding boxes",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    Verovio.configure(verovio_path=args.verovio_path)
    Inkscape.configure(inkscape_path=args.inkscape_path)
    logging.basicConfig(level=args.logging_level)

    logging.info("Checking that the input path exists")
    if not args.musicxml_path.exists():
        logging.error("The provided input path does not exist")
        exit(1)
    logging.info("Creating output path")
    args.clean_path.mkdir(exist_ok=True, parents=False)
    args.augmented_path.mkdir(exist_ok=True, parents=False)

    if not args.omit_musicxml_generation:
        generate_musicxmls(args.musicxml_path, args.clean_path)
    render(args.clean_path, args.augmented_path, args.create_debug_images)


if __name__ == "__main__":
    main(setup())
