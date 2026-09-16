# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ocr-reco (3.11.14.final.0)
#     language: python
#     name: python3
# ---

#
# # %%
from typing import TypeVar, Any
from pathlib import Path
import smashcima as sc
import numpy as np
import cv2
from itertools import chain

T = TypeVar("T", bound="sc.AffineSpaceVisitor")


def make_scene_to_canvas_transform(
    view_box: "sc.ViewBox", dpi: float
) -> "sc.Transform":
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
        """Creates the visitor instance for a sub space"""
        return GlyphVisitor(sub_space, self.root_space, self.scene_to_canvas_transform)

    def accept_sub_visitor(self, sub_visitor: "GlyphVisitor") -> None:
        """Once sub space visiting finished, incorporate its results"""
        self.subelements = {
            k: v
            for k, v in chain(self.subelements.items(), sub_visitor.subelements.items())
        }

    def visit_scene_object(self, obj: sc.SceneObject):
        """Handle scene objects that are children but are not affine spaces"""
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


OVERWRITE_EXISTING = False
"""Set to True to re-render samples even if their output files already exist."""


def sample_exists(musicxml_path: Path, output_path: Path) -> bool:
    """True if both the PNG and the bbox JSON for this sample are on disk."""
    return (output_path / musicxml_path.with_suffix(".png").name).exists() and (
        output_path / musicxml_path.with_suffix(".json").name
    ).exists()


def draw_bboxes(
    bitmap: np.ndarray,
    bboxes: dict[str, dict],
    color: tuple[int, int, int] = (0, 255, 0),
    staffline_color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draws the bboxes returned by MppModel.__call__ on top of the bitmap
    it was returned alongside, for visual debugging."""
    canvas = bitmap[:, :, :3].copy() if bitmap.shape[2] == 4 else bitmap.copy()

    for bbox in bboxes.values():
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        box_color = (
            staffline_color if bbox["smufl_id"] == "smashcima::staffLine" else color
        )
        cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, thickness)

    return canvas


class MppModel(sc.orchestration.BaseHandwrittenModel):
    """Model that synthesizes piano staves with MUSCIMA++ glyphs"""

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
        file: Path | str,
        aspect_ratio_hint: float,
        # sample_seed: int,
        full_resolution: bool,
        no_augmentation: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        :param file: Path to the .musicxml file to synthesize
        :param grandstaff_aspect_ratio: width/height aspect ratio
            of the original grandstaff sample image.
        """
        # set rng seed for this synthetic sample
        # (and for cv2 and numpy as well since postprocessing filters
        # don't respect the random.Random instance in smashcima)
        # self.rng.seed(sample_seed)

        # also set seed on global generators to make sure there is
        # absolutely no uncontrolled randomness
        # random.seed(sample_seed)
        # np.random.seed(sample_seed % (2**31))
        # cv2.setRNGSeed(sample_seed % (2**31))

        # configure page size and layout
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

        # scale down the image to reduce dataset size on disk and in RAM
        # scale down to height of 192, which is exactly what the model wants
        if not full_resolution:
            scale_ratio = 128 / bitmap.shape[0]
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


# %%
from pathlib import Path  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402

demo_file = Path("/DATA/MonophonicLines/w0_p0_m6to12_v5.musicxml")
demo_path = Path("./demo.png")

synthesis_model = MppModel(aspect_ratio_magic_factor=1)
sample, bboxes = synthesis_model(
    file=demo_file,
    aspect_ratio_hint=256 / 64,
    # sample_seed=3,
    full_resolution=True,
    no_augmentation=True,
)

plt.figure()
plt.imshow(
    cv2.cvtColor(
        draw_bboxes(sample, bboxes),
        cv2.COLOR_BGR2RGB,
    )
)
plt.show()
plt.close()

draw_bboxes(sample, bboxes)
# %%
from tqdm.auto import tqdm
import json

MONOPHONIC_LINE_PATH = Path("/DATA/MonophonicLines/")
MONOPHONIC_LINE_OUTPUT_PATH = Path("/DATA/MonophonicLinesSmashcima/")


MONOPHONIC_LINE_OUTPUT_PATH.mkdir(exist_ok=True)


synthesis_model = MppModel(aspect_ratio_magic_factor=1)
for musicxml_path in tqdm(MONOPHONIC_LINE_PATH.glob("*.musicxml")):
    if not OVERWRITE_EXISTING and sample_exists(
        musicxml_path, MONOPHONIC_LINE_OUTPUT_PATH
    ):
        continue

    try:
        sample, bboxes = synthesis_model(
            file=musicxml_path,
            aspect_ratio_hint=256 / 64,
            full_resolution=True,
            no_augmentation=True,
        )
        cv2.imwrite(
            str(MONOPHONIC_LINE_OUTPUT_PATH / musicxml_path.with_suffix(".png").name),
            sample,
        )
        with open(
            MONOPHONIC_LINE_OUTPUT_PATH / musicxml_path.with_suffix(".json").name, "w"
        ) as f_json:
            json.dump(bboxes, f_json, indent=4)
    except Exception:
        print(f"Line {musicxml_path} could not be rendered. Skipping...")
        continue

# %%
from multiprocessing import get_context  # noqa: E402

N_WORKERS = 8

MONOPHONIC_LINE_PATH = Path("/DATA/MonophonicLines/")
MONOPHONIC_LINE_OUTPUT_PATH = Path("/DATA/MonophonicLinesSmashcima/")

_worker_model = None


def _init_worker():
    global _worker_model
    # cv2/BLAS thread pools survive the fork; 8 processes each spawning
    # their own internal threads oversubscribes the CPU and can deadlock
    # right after fork, so pin every worker to a single thread
    cv2.setNumThreads(1)
    _worker_model = MppModel(aspect_ratio_magic_factor=1)


def _render_one(musicxml_path: Path) -> tuple[Path, bool]:
    if not OVERWRITE_EXISTING and sample_exists(
        musicxml_path, MONOPHONIC_LINE_OUTPUT_PATH
    ):
        return musicxml_path, True

    try:
        sample, bboxes = _worker_model(
            file=musicxml_path,
            aspect_ratio_hint=256 / 64,
            full_resolution=True,
            no_augmentation=True,
        )
        cv2.imwrite(
            str(MONOPHONIC_LINE_OUTPUT_PATH / musicxml_path.with_suffix(".png").name),
            sample,
        )
        with open(
            MONOPHONIC_LINE_OUTPUT_PATH / musicxml_path.with_suffix(".json").name, "w"
        ) as f_json:
            json.dump(bboxes, f_json, indent=4)
        return musicxml_path, True
    except Exception:
        return musicxml_path, False


MONOPHONIC_LINE_OUTPUT_PATH.mkdir(exist_ok=True)
musicxml_paths = list(MONOPHONIC_LINE_PATH.glob("*.musicxml"))

cv2.setNumThreads(1)
with get_context("fork").Pool(N_WORKERS, initializer=_init_worker) as pool:
    for musicxml_path, ok in tqdm(
        pool.imap_unordered(_render_one, musicxml_paths),
        total=len(musicxml_paths),
    ):
        if not ok:
            print(f"Line {musicxml_path} could not be rendered. Skipping...")
