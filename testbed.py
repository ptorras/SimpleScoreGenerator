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


def turn_measure_monophonic(measure: m21.stream.Measure) -> m21.stream.Measure:
    voices = measure.getElementsByClass(m21.stream.Voice)
    if len(voices) > 1:
        measure = measure.cloneEmpty()
        for elm in voices[0]:
            measure.append(elm)
    return measure


def turn_score_monophonic(score: m21.stream.Score) -> m21.stream.Score:
    out_score = score.cloneEmpty()
    for part in score.parts:
        newpart = part.cloneEmpty()
        for measure in part.getElementsByClass(m21.stream.Measure):
            newpart.append(deepcopy(turn_measure_monophonic(measure)))
        out_score.append(newpart)
    return out_score
