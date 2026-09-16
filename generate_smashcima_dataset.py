"""Batch-renders monophonic MusicXML lines into single-staff Smashcima images,
alongside JSON files with glyph and staff-line bounding boxes.
"""

import argparse
import json
import zipfile
from contextlib import contextmanager
from itertools import chain
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional

import cv2
import numpy as np
import smashcima as sc
from tqdm.auto import tqdm

ASPECT_RATIO_HINT = 256 / 64


def make_scene_to_canvas_transform(view_box: "sc.ViewBox", dpi: float) -> "sc.Transform":
    """Builds the same mm -> pixel transform used internally by Smashcima's
    DefaultCompositor when it rasterizes a page (see
    smashcima/exporting/compositing/DefaultCompositor.py:extract_layers).

    `AffineSpace.transform_from(obj.space)` only gets you into the *root*
    affine space, which is still in millimeters and still offset by the
    page's own placement (`view_box.rectangle.top_left_corner`). You need to
    additionally subtract that offset and scale by the DPI to land in the
    same pixel grid as `scene.render(page)`.
    """
    return sc.Transform.translate(-view_box.rectangle.top_left_corner.vector).then(
        sc.Transform.scale(sc.mm_to_px(1, dpi=dpi))
    )


class GlyphVisitor(sc.AffineSpaceVisitor):
    """Walks the scene hierarchy and collects glyph bounding boxes in pixel
    space, aligned with the bitmap produced by `scene.render(page)`."""

    def __init__(
        self,
        space: sc.AffineSpace,
        root_space: sc.AffineSpace,
        scene_to_canvas_transform: "sc.Transform",
    ) -> None:
        super().__init__(space)

        self.subelements = {}
        self.root_space = root_space
        self.scene_to_canvas_transform = scene_to_canvas_transform

    def create_sub_visitor(self, sub_space: sc.AffineSpace) -> "GlyphVisitor":
        return GlyphVisitor(sub_space, self.root_space, self.scene_to_canvas_transform)

    def accept_sub_visitor(self, sub_visitor: "GlyphVisitor") -> None:
        self.subelements = {
            k: v
            for k, v in chain(self.subelements.items(), sub_visitor.subelements.items())
        }

    def visit_scene_object(self, obj: sc.SceneObject):
        # ComposedGlyph is a Glyph too (e.g. the "smashcima::staff" glyph that
        # groups all 5 staffline glyphs) - skip it here, its leaf sub_glyphs
        # are visited on their own and are what we actually want bboxes for.
        if isinstance(obj, sc.Glyph) and not isinstance(obj, sc.ComposedGlyph):
            self.subelements["obj_" + str(id(obj))] = glyph_pixel_bbox(
                obj, self.root_space, self.scene_to_canvas_transform
            )


def glyph_pixel_bbox(
    obj: "sc.Glyph",
    root_space: sc.AffineSpace,
    scene_to_canvas_transform: "sc.Transform",
) -> dict:
    """Computes a glyph's bounding box in pixel coordinates that match the
    bitmap produced by `scene.render(page)` (before any post-hoc cropping)."""
    contours_in_root = obj.region.get_contours_in_space(root_space)
    contours_in_canvas = scene_to_canvas_transform.apply_to(contours_in_root)
    bbox = contours_in_canvas.bbox()
    return {
        "x": bbox.x,
        "y": bbox.y,
        "w": bbox.width,
        "h": bbox.height,
        "smufl_id": obj.label,
    }


def sample_exists(musicxml_path: Path, output_path: Path) -> bool:
    """True if both the PNG and the bbox JSON for this sample are on disk."""
    return (output_path / musicxml_path.with_suffix(".png").name).exists() and (
        output_path / musicxml_path.with_suffix(".json").name
    ).exists()


class MppModel(sc.orchestration.BaseHandwrittenModel):
    """Model that synthesizes single-staff lines with MUSCIMA++ glyphs"""

    def __init__(self, aspect_ratio_magic_factor: float):
        super().__init__()
        self.aspect_ratio_magic_factor = aspect_ratio_magic_factor

    def resolve_services(self) -> None:
        super().resolve_services()

        # page synthesizer is actually the SimplePageSynthesizer
        self.page_synthesizer: sc.synthesis.SimplePageSynthesizer = (
            self.page_synthesizer
        )

        # layout synthesizer is actually ColumnMusicNotationSynthesizer
        self.notation_synthesizer: sc.synthesis.ColumnMusicNotationSynthesizer = (
            self.notation_synthesizer
        )

    def configure_services(self):
        super().configure_services()

        # disable system breaks all together
        self.notation_synthesizer.disable_wrapping = True
        self.notation_synthesizer.respect_line_and_page_breaks = False

    def __call__(
        self,
        file: Path,
        aspect_ratio_hint: float,
        target_height: Optional[int],
        no_augmentation: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        :param file: Path to the .musicxml file to synthesize
        :param aspect_ratio_hint: width/height aspect ratio of the page
        :param target_height: if set, the rendered image (and its bboxes)
            are rescaled to this height in pixels; if None, full resolution
            is kept
        """
        staff_height = self.page_synthesizer.stafflines_synthesizer.staff_height

        ps = self.page_synthesizer.page_setup
        ps.padding_top = staff_height * 2
        ps.padding_bottom = staff_height * 2
        ps.padding_left = staff_height * 2
        ps.padding_right = staff_height * 2
        ps.staff_count = 1

        page_height = staff_height * 6
        ps.size = sc.Vector2(
            aspect_ratio_hint * page_height * self.aspect_ratio_magic_factor,
            page_height,
        )

        # stretch out notation to full width in 50% of cases
        self.notation_synthesizer.stretch_out_columns = self.rng.random() < 0.5

        # run synthesis
        scene = super().__call__(file=file)
        assert len(scene.pages) == 1

        # configure postprocessing (image augmentations)
        if not no_augmentation:
            pp = sc.BaseHandwrittenPostprocessor(self.rng)
            pp.f_scribbles.force_dont = True  # handling fonts is a mess
            scene.compositor = sc.DefaultCompositor(pp)

        # rasterize scene
        page = scene.pages[0]
        bitmap = scene.render(page)

        # the same mm -> pixel transform the compositor used to build `bitmap`
        scene_to_canvas_transform = make_scene_to_canvas_transform(
            view_box=page.view_box, dpi=scene.dpi
        )

        # Visit scene to get the glyph bounding boxes, already in the
        # same pixel space as `bitmap`
        visitor = GlyphVisitor(
            scene.root_space, scene.root_space, scene_to_canvas_transform
        )
        visitor.run()
        subelements = visitor.subelements

        # staff lines aren't picked up by GlyphVisitor: the individual
        # staffline glyphs live as sub_glyphs of the ComposedGlyph
        # ("smashcima::staff") attached to each StaffVisual, and
        # ComposedGlyph is intentionally skipped above
        for staff in page.staves:
            for line_glyph in staff.glyph.sub_glyphs:
                subelements["obj_" + str(id(line_glyph))] = glyph_pixel_bbox(
                    line_glyph, scene.root_space, scene_to_canvas_transform
                )

        # shrink view box by one staff height
        shrink_pixels = int(sc.mm_to_px(staff_height, dpi=scene.dpi))
        bitmap = bitmap[shrink_pixels:-shrink_pixels, shrink_pixels:-shrink_pixels, :]

        # the crop above shifted the pixel origin: offset every bbox to
        # match, then drop anything that no longer overlaps the crop
        cropped_height, cropped_width = bitmap.shape[:2]
        visible_subelements = {}
        for key, bbox in subelements.items():
            x = bbox["x"] - shrink_pixels
            y = bbox["y"] - shrink_pixels
            x0, y0 = max(x, 0), max(y, 0)
            x1 = min(x + bbox["w"], cropped_width)
            y1 = min(y + bbox["h"], cropped_height)
            if x1 <= x0 or y1 <= y0:
                continue  # bbox fell entirely outside the cropped image
            visible_subelements[key] = {
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "smufl_id": bbox["smufl_id"],
            }
        subelements = visible_subelements

        # scale to the target height to reduce dataset size on disk and in RAM
        if target_height is not None:
            scale_ratio = target_height / bitmap.shape[0]
            bitmap = cv2.resize(
                bitmap,
                dsize=(
                    int(bitmap.shape[1] * scale_ratio),
                    int(bitmap.shape[0] * scale_ratio),
                ),
                interpolation=cv2.INTER_AREA,
            )
            for bbox in subelements.values():
                bbox["x"] *= scale_ratio
                bbox["y"] *= scale_ratio
                bbox["w"] *= scale_ratio
                bbox["h"] *= scale_ratio

        return bitmap, subelements


def render_one(
    musicxml_path: Path,
    model: MppModel,
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
) -> tuple[Path, bool]:
    """Renders a single MusicXML file and writes its PNG/JSON outputs.
    Returns (musicxml_path, success)."""
    if not overwrite and sample_exists(musicxml_path, output_path):
        return musicxml_path, True

    try:
        sample, bboxes = model(
            file=musicxml_path,
            aspect_ratio_hint=ASPECT_RATIO_HINT,
            target_height=target_height,
            no_augmentation=True,
        )
    except Exception:
        return musicxml_path, False

    cv2.imwrite(str(output_path / musicxml_path.with_suffix(".png").name), sample)
    with open(output_path / musicxml_path.with_suffix(".json").name, "w") as f_json:
        json.dump(bboxes, f_json, indent=4)

    return musicxml_path, True


def run_serial(
    musicxml_paths: list[Path],
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
) -> None:
    model = MppModel(aspect_ratio_magic_factor=1)
    for musicxml_path in tqdm(musicxml_paths):
        _, ok = render_one(musicxml_path, model, output_path, target_height, overwrite)
        if not ok:
            print(f"Line {musicxml_path} could not be rendered. Skipping...")


_worker_state: dict[str, Any] = {}


def _init_worker(output_path: Path, target_height: Optional[int], overwrite: bool) -> None:
    # cv2/BLAS thread pools survive the fork; N processes each spawning
    # their own internal threads oversubscribes the CPU and can deadlock
    # right after fork, so pin every worker to a single thread
    cv2.setNumThreads(1)
    _worker_state["model"] = MppModel(aspect_ratio_magic_factor=1)
    _worker_state["output_path"] = output_path
    _worker_state["target_height"] = target_height
    _worker_state["overwrite"] = overwrite


def _render_one_in_worker(musicxml_path: Path) -> tuple[Path, bool]:
    return render_one(
        musicxml_path,
        _worker_state["model"],
        _worker_state["output_path"],
        _worker_state["target_height"],
        _worker_state["overwrite"],
    )


def run_parallel(
    musicxml_paths: list[Path],
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
    workers: int,
) -> None:
    cv2.setNumThreads(1)
    with get_context("fork").Pool(
        workers,
        initializer=_init_worker,
        initargs=(output_path, target_height, overwrite),
    ) as pool:
        for musicxml_path, ok in tqdm(
            pool.imap_unordered(_render_one_in_worker, musicxml_paths),
            total=len(musicxml_paths),
        ):
            if not ok:
                print(f"Line {musicxml_path} could not be rendered. Skipping...")


@contextmanager
def resolve_musicxml_root(root: Path) -> Iterator[list[Path]]:
    """Yields the sorted list of .musicxml files under `root`.

    If `root` is a .zip file, it is extracted into a temporary directory
    (cleaned up on exit) and every .musicxml file inside it, at any depth,
    is picked up. Otherwise `root` is treated as a directory and searched
    for .musicxml files at its top level.
    """
    if root.is_file() and root.suffix.lower() == ".zip":
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(root) as zf:
                zf.extractall(tmp_path)
            yield sorted(tmp_path.rglob("*.musicxml"))
    else:
        yield sorted(root.glob("*.musicxml"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing the input .musicxml files, or a .zip "
        "archive containing them (searched at any depth)",
    )
    parser.add_argument(
        "output", type=Path, help="Directory to write the rendered samples to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes to render with. 0 disables "
        "multiprocessing and renders on the main process (default: 8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render samples even if their output files already exist "
        "(default: skip them)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Target image height in pixels. If unset, images are kept at "
        "full resolution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with resolve_musicxml_root(args.root) as musicxml_paths:
        if args.workers == 0:
            run_serial(musicxml_paths, args.output, args.resolution, args.overwrite)
        else:
            run_parallel(
                musicxml_paths,
                args.output,
                args.resolution,
                args.overwrite,
                args.workers,
            )


if __name__ == "__main__":
    main()
