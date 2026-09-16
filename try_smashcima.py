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

# %%
from typing import TypeVar, Any
from pathlib import Path
import smashcima as sc
import numpy as np
import cv2
import random
from itertools import chain

T = TypeVar("T", bound="sc.AffineSpaceVisitor")


class GlyphVisitor(sc.AffineSpaceVisitor):
    """Base class for walking through the scene hierarchy"""

    def __init__(self, space: sc.AffineSpace, root_space: sc.AffineSpace) -> None:
        super().__init__(space)

        self.subelements = {}
        self.root_space = root_space

    def create_sub_visitor(self, sub_space: sc.AffineSpace) -> "GlyphVisitor":
        """Creates the visitor instance for a sub space"""
        return GlyphVisitor(sub_space, self.root_space)

    def accept_sub_visitor(self, sub_visitor: "GlyphVisitor") -> None:
        """Once sub space visiting finished, incorporate its results"""
        self.subelements = {
            k: v
            for k, v in chain(self.subelements.items(), sub_visitor.subelements.items())
        }

    def visit_scene_object(self, obj: sc.SceneObject):
        """Handle scene objects that are children but are not affine spaces"""
        if isinstance(obj, sc.Glyph):
            bbox = obj.get_bbox_in_space(self.root_space)
            label = obj.label
            self.subelements["obj_" + str(id(obj))] = {
                "x": bbox.x,
                "y": bbox.y,
                "w": bbox.width,
                "h": bbox.height,
                "smufl_id": label,
            }


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
        sample_seed: int,
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
        self.rng.seed(sample_seed)

        # also set seed on global generators to make sure there is
        # absolutely no uncontrolled randomness
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**31))
        cv2.setRNGSeed(sample_seed % (2**31))

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
        bitmap = scene.render(scene.pages[0])

        # Visit scene to get the bounding boxes
        visitor = GlyphVisitor(scene.root_space, scene.root_space)
        visitor.run()

        # shrink view box by one staff height
        shrink_pixels = int(sc.mm_to_px(staff_height, dpi=scene.dpi))
        bitmap = bitmap[shrink_pixels:-shrink_pixels, shrink_pixels:-shrink_pixels, :]

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

        return bitmap, visitor.subelements


# %%
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

demo_file = Path("/DATA/MonophonicLines/w0_p0_m6to12_v5.musicxml")
demo_path = Path("./demo.png")

synthesis_model = MppModel(aspect_ratio_magic_factor=1)
sample, bboxes = synthesis_model(
    file=demo_file,
    aspect_ratio_hint=256 / 64,
    sample_seed=3,
    full_resolution=True,
    no_augmentation=True,
)

plt.figure()
plt.imshow(
    cv2.cvtColor(
        sample,
        cv2.COLOR_BGR2RGB,
    )
)
plt.show()
plt.close()

# %%
