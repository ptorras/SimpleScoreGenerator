# %%
from copy import deepcopy
from pathlib import Path

import music21 as m21

from proc.drivers import Inkscape, MuseScore, Verovio

Verovio.configure(verovio_path=Path("verovio"))
Inkscape.configure(
    inkscape_path=Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")
)
MuseScore.configure(
    mscore_path=Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
)

DATASET_PATH = Path("~/Documents/Datasets/ComrefMusicxml")
EXAMPLE_FILE = DATASET_PATH / "sq8938822.mxl"

score = m21.converter.parse(EXAMPLE_FILE)
# %%
