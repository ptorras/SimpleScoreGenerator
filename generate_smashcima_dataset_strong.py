"""Batch-renders monophonic MusicXML lines into single-staff Smashcima images,
alongside JSON files with glyph bounding boxes and staff-line endpoints, and
.author files with the MUSCIMA++ writer number whose handwriting glyphs were
used.

This is a higher-variability variant of `generate_smashcima_dataset.py`: it
swaps in `WavyStafflinesSynthesizer` (staff lines with variable stroke width
and gentle waviness instead of straight uniform rectangles) and
`StrongHandwrittenPostprocessor` (Kanungo-style noise, foreground
dilation/erosion, extra background noise/text, an extra affine jitter, and
an independent ink-layer jitter - see `smashcima_strong_augmentations.py`),
and randomizes the crop margin above/below the staff instead of using a
fixed one. Augmentation is applied by default (unlike the base script).

Every filter that warps geometry (rotation, the extra affine jitter, the
ink-layer jitter) also updates the glyph/staffline pixel coordinates, so the
JSON bboxes stay correct under augmentation instead of only reflecting the
scene's pre-augmentation vector geometry. Musical glyphs are still reported
as an axis-aligned box (`x`/`y`/`w`/`h`); staff lines are instead reported as
their 2 centerline endpoints (`x1`/`y1`/`x2`/`y2`), since an axis-aligned box
around a rotated line balloons badly (a long thin rectangle's true rotated
bbox is a much taller parallelogram).
"""

import argparse
import json
import shutil
import zipfile
from contextlib import contextmanager
from itertools import chain
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional, Tuple

import cv2
import numpy as np
import smashcima as sc
from tqdm.auto import tqdm

from smashcima_strong_augmentations import (
    StrongHandwrittenPostprocessor,
    WavyStafflinesSynthesizer,
)

ASPECT_RATIO_HINT = 256 / 64


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


def bbox_corner_points(bbox: dict) -> list[Tuple[float, float]]:
    """The 4 corners of a `glyph_pixel_bbox`-style bbox dict, for tracking a
    glyph through the postprocessor as an axis-aligned box: once warping is
    done, the box is recovered as the enclosing box of these corners."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def bbox_line_endpoints(bbox: dict) -> list[Tuple[float, float]]:
    """The 2 endpoints (left, right) of a `glyph_pixel_bbox`-style bbox
    dict's vertical midline, for tracking a staffline through the
    postprocessor as a line segment rather than an axis-aligned box: an
    axis-aligned box around a rotated line balloons badly, since a long
    thin rectangle's true rotated bbox is a much taller parallelogram."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    mid_y = y + h / 2
    return [(x, mid_y), (x + w, mid_y)]


def clip_segment_to_box(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    width: float,
    height: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clips the (p1, p2) line segment to the [0, width] x [0, height] box
    using the Liang-Barsky algorithm, returning the visible sub-segment's
    endpoints, or None if the segment doesn't intersect the box at all."""
    x1, y1 = p1
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]

    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1),
        (dx, width - x1),
        (-dy, y1),
        (dy, height - y1),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return None

    return (
        (x1 + t0 * dx, y1 + t0 * dy),
        (x1 + t1 * dx, y1 + t1 * dy),
    )


def output_stem_path(
    musicxml_path: Path, musicxml_root: Path, output_path: Path
) -> Path:
    """Maps an input .musicxml file to its output path (without suffix),
    mirroring its location relative to `musicxml_root` so that files with
    the same basename in different subdirectories don't collide."""
    return output_path / musicxml_path.relative_to(musicxml_root).with_suffix("")


def sample_exists(musicxml_path: Path, musicxml_root: Path, output_path: Path) -> bool:
    """True if the JPG, the bbox JSON, the copied MusicXML, and the author
    file for this sample are all on disk."""
    stem_path = output_stem_path(musicxml_path, musicxml_root, output_path)
    return (
        stem_path.with_suffix(".jpg").exists()
        and stem_path.with_suffix(".json").exists()
        and stem_path.with_suffix(musicxml_path.suffix).exists()
        and stem_path.with_suffix(".author").exists()
    )


class MppModel(sc.orchestration.BaseHandwrittenModel):
    """Model that synthesizes single-staff lines with MUSCIMA++ glyphs,
    using stronger and more varied augmentations than the base MppModel:
    wavy/variable-width stafflines, a heavier postprocessing pipeline, and
    a randomized crop margin above/below the staff."""

    # crop margin above/below the staff, as a multiple of staff height,
    # sampled independently per sample and per side
    VERTICAL_CROP_RANGE = (0.4, 1.6)

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

        # swap in the wavy/variable-width stafflines synthesizer
        self.page_synthesizer.stafflines_synthesizer = WavyStafflinesSynthesizer(
            self.rng
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
    ) -> tuple[np.ndarray, dict[str, Any], int]:
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
        page = scene.pages[0]

        # the same mm -> pixel transform the compositor will use to build
        # the rendered bitmap
        scene_to_canvas_transform = make_scene_to_canvas_transform(
            view_box=page.view_box, dpi=scene.dpi
        )

        # glyph bounding boxes in the PRE-postprocessing pixel space (this
        # only depends on the scene's vector geometry, not on rendering);
        # corrected below to also reflect any geometry-warping
        # augmentations (rotation, affine jitter, ink-layer jitter)
        visitor = GlyphVisitor(
            scene.root_space, scene.root_space, scene_to_canvas_transform
        )
        visitor.run()
        ink_subelements = visitor.subelements

        # staff lines aren't picked up by GlyphVisitor: the individual
        # staffline glyphs live as sub_glyphs of the ComposedGlyph
        # ("smashcima::staff") attached to each StaffVisual, and
        # ComposedGlyph is intentionally skipped above
        staffline_subelements = {}
        for staff in page.staves:
            for line_glyph in staff.glyph.sub_glyphs:
                staffline_subelements["obj_" + str(id(line_glyph))] = (
                    glyph_pixel_bbox(
                        line_glyph, scene.root_space, scene_to_canvas_transform
                    )
                )

        # glyphs are tracked as their bbox's 4 corners (collapsed back into
        # a box below); stafflines are tracked as just their 2 centerline
        # endpoints and stay a line segment, never collapsed into a box
        # (see `bbox_line_endpoints`)
        ink_points = [bbox_corner_points(b) for b in ink_subelements.values()]
        staffline_points = [
            bbox_line_endpoints(b) for b in staffline_subelements.values()
        ]

        # configure postprocessing (image augmentations)
        pp = None
        if not no_augmentation:
            pp = StrongHandwrittenPostprocessor(self.rng)
            pp.f_scribbles.force_dont = True  # handling fonts is a mess
            pp.tracked_ink_points = ink_points
            pp.tracked_staffline_points = staffline_points
            scene.compositor = sc.DefaultCompositor(pp)

        # rasterize scene; this runs the postprocessing pipeline, which -
        # when augmentation is enabled - mutates pp.tracked_ink_points,
        # pp.tracked_staffline_points and pp.tracked_points in place to
        # follow every geometric warp it applies
        bitmap = scene.render(page)

        if pp is not None:
            ink_points = pp.tracked_points[: len(ink_subelements)]
            staffline_points = pp.tracked_points[len(ink_subelements) :]

        # write the (possibly warped) points back onto their glyphs: ink
        # glyphs keep the x/y/w/h bbox format (recovered as the enclosing
        # box of the 4 corners), staffline glyphs switch to an x1/y1/x2/y2
        # line-segment format instead
        for key, corners in zip(ink_subelements, ink_points):
            b = ink_subelements[key]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            b["x"], b["y"] = x1, y1
            b["w"], b["h"] = max(x2 - x1, 0.0), max(y2 - y1, 0.0)

        for key, (p1, p2) in zip(staffline_subelements, staffline_points):
            b = staffline_subelements[key]
            del b["x"], b["y"], b["w"], b["h"]
            b["x1"], b["y1"] = p1
            b["x2"], b["y2"] = p2

        subelements = {**ink_subelements, **staffline_subelements}

        # shrink view box by one staff height on the sides, and by a
        # randomized, independent amount above/below the staff (instead of
        # a fixed staff height on every side) for cropping variability
        side_shrink_pixels = int(sc.mm_to_px(staff_height, dpi=scene.dpi))
        top_shrink_pixels = int(
            sc.mm_to_px(
                staff_height * self.rng.uniform(*self.VERTICAL_CROP_RANGE),
                dpi=scene.dpi,
            )
        )
        bottom_shrink_pixels = int(
            sc.mm_to_px(
                staff_height * self.rng.uniform(*self.VERTICAL_CROP_RANGE),
                dpi=scene.dpi,
            )
        )
        bitmap = bitmap[
            top_shrink_pixels : bitmap.shape[0] - bottom_shrink_pixels,
            side_shrink_pixels:-side_shrink_pixels,
            :,
        ]

        # the crop above shifted the pixel origin: offset every glyph/line
        # to match, then drop anything that no longer overlaps the crop
        cropped_height, cropped_width = bitmap.shape[:2]
        visible_subelements = {}
        for key, elem in subelements.items():
            if "x1" in elem:
                # staffline: clip the line segment to the cropped area
                p1 = (elem["x1"] - side_shrink_pixels, elem["y1"] - top_shrink_pixels)
                p2 = (elem["x2"] - side_shrink_pixels, elem["y2"] - top_shrink_pixels)
                clipped = clip_segment_to_box(p1, p2, cropped_width, cropped_height)
                if clipped is None:
                    continue  # line fell entirely outside the cropped image
                (x1, y1), (x2, y2) = clipped
                visible_subelements[key] = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "smufl_id": elem["smufl_id"],
                }
            else:
                # musical glyph: clip the bbox to the cropped area
                x = elem["x"] - side_shrink_pixels
                y = elem["y"] - top_shrink_pixels
                x0, y0 = max(x, 0), max(y, 0)
                x1 = min(x + elem["w"], cropped_width)
                y1 = min(y + elem["h"], cropped_height)
                if x1 <= x0 or y1 <= y0:
                    continue  # bbox fell entirely outside the cropped image
                visible_subelements[key] = {
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "smufl_id": elem["smufl_id"],
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
            for elem in subelements.values():
                if "x1" in elem:
                    elem["x1"] *= scale_ratio
                    elem["y1"] *= scale_ratio
                    elem["x2"] *= scale_ratio
                    elem["y2"] *= scale_ratio
                else:
                    elem["x"] *= scale_ratio
                    elem["y"] *= scale_ratio
                    elem["w"] *= scale_ratio
                    elem["h"] *= scale_ratio

        return bitmap, subelements, scene.mpp_writer


def render_one(
    musicxml_path: Path,
    musicxml_root: Path,
    model: MppModel,
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
    no_augmentation: bool,
) -> tuple[Path, bool, str]:
    """Renders a single MusicXML file and writes its JPG/JSON outputs.
    Returns (musicxml_path, success)."""
    if not overwrite and sample_exists(musicxml_path, musicxml_root, output_path):
        return musicxml_path, True, ""

    try:
        sample, bboxes, writer = model(
            file=musicxml_path,
            aspect_ratio_hint=ASPECT_RATIO_HINT,
            target_height=target_height,
            no_augmentation=no_augmentation,
        )
    except Exception as e:
        return musicxml_path, False, str(e)

    stem_path = output_stem_path(musicxml_path, musicxml_root, output_path)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stem_path.with_suffix(".jpg")), sample)
    with open(stem_path.with_suffix(".json"), "w") as f_json:
        json.dump(bboxes, f_json, indent=4)
    shutil.copy2(musicxml_path, stem_path.with_suffix(musicxml_path.suffix))
    stem_path.with_suffix(".author").write_text(str(writer))

    return musicxml_path, True, ""


def run_serial(
    musicxml_paths: list[Path],
    musicxml_root: Path,
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
    no_augmentation: bool,
) -> None:
    model = MppModel(aspect_ratio_magic_factor=1)
    for musicxml_path in tqdm(musicxml_paths):
        _, ok, err = render_one(
            musicxml_path,
            musicxml_root,
            model,
            output_path,
            target_height,
            overwrite,
            no_augmentation,
        )
        if not ok:
            print(f"Line {musicxml_path} could not be rendered: {err}. Skipping...")


_worker_state: dict[str, Any] = {}


def _init_worker(
    musicxml_root: Path,
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
    no_augmentation: bool,
) -> None:
    # cv2/BLAS thread pools survive the fork; N processes each spawning
    # their own internal threads oversubscribes the CPU and can deadlock
    # right after fork, so pin every worker to a single thread
    cv2.setNumThreads(1)
    _worker_state["model"] = MppModel(aspect_ratio_magic_factor=1)
    _worker_state["musicxml_root"] = musicxml_root
    _worker_state["output_path"] = output_path
    _worker_state["target_height"] = target_height
    _worker_state["overwrite"] = overwrite
    _worker_state["no_augmentation"] = no_augmentation


def _render_one_in_worker(musicxml_path: Path) -> tuple[Path, bool, str]:
    return render_one(
        musicxml_path,
        _worker_state["musicxml_root"],
        _worker_state["model"],
        _worker_state["output_path"],
        _worker_state["target_height"],
        _worker_state["overwrite"],
        _worker_state["no_augmentation"],
    )


def run_parallel(
    musicxml_paths: list[Path],
    musicxml_root: Path,
    output_path: Path,
    target_height: Optional[int],
    overwrite: bool,
    workers: int,
    no_augmentation: bool,
) -> None:
    cv2.setNumThreads(1)
    with get_context("fork").Pool(
        workers,
        initializer=_init_worker,
        initargs=(
            musicxml_root,
            output_path,
            target_height,
            overwrite,
            no_augmentation,
        ),
    ) as pool:
        for musicxml_path, ok, err in tqdm(
            pool.imap_unordered(_render_one_in_worker, musicxml_paths),
            total=len(musicxml_paths),
        ):
            if not ok:
                print(f"Line {musicxml_path} could not be rendered: {err}. Skipping...")


@contextmanager
def resolve_musicxml_root(root: Path) -> Iterator[tuple[Path, list[Path]]]:
    """Yields (search_root, sorted list of .musicxml files under it).

    If `root` is a .zip file, it is extracted into a temporary directory
    (cleaned up on exit) and every .musicxml file inside it, at any depth,
    is picked up; `search_root` is that temporary directory, so callers can
    recover each file's path relative to the archive and mirror the
    archive's directory structure in the output (files with the same
    basename in different subdirectories must not collide). Otherwise
    `root` is treated as a directory and searched for .musicxml files at
    its top level.
    """
    if root.is_file() and root.suffix.lower() == ".zip":
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(root) as zf:
                zf.extractall(tmp_path)
            yield tmp_path, sorted(tmp_path.rglob("*.musicxml"))
    else:
        yield root, sorted(root.glob("*.musicxml"))


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
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable the postprocessing augmentation pipeline (default: "
        "enabled, unlike generate_smashcima_dataset.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with resolve_musicxml_root(args.root) as (musicxml_root, musicxml_paths):
        if args.workers == 0:
            run_serial(
                musicxml_paths,
                musicxml_root,
                args.output,
                args.resolution,
                args.overwrite,
                args.no_augmentation,
            )
        else:
            run_parallel(
                musicxml_paths,
                musicxml_root,
                args.output,
                args.resolution,
                args.overwrite,
                args.workers,
                args.no_augmentation,
            )


if __name__ == "__main__":
    main()
