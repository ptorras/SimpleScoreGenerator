#  Copyright (c) 2026 Pau Torras.
#
#  This program is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this
#  program. If not, see <https://www.gnu.org/licenses/>.

import json
import re
from pathlib import Path
from typing import Any

_METADATA_PATH = Path(__file__).parent / "smufl_metadata"

with (
    open(_METADATA_PATH / "classes.json", "r") as f_classes,
    open(_METADATA_PATH / "glyphnames.json", "r") as f_glyphnames,
    open(_METADATA_PATH / "ranges.json", "r") as f_ranges,
):
    _DATA_CLASSES = json.load(f_classes)
    _DATA_GLYPHNAMES: dict[str, dict[str, Any]] = json.load(f_glyphnames)
    _DATA_RANGES = json.load(f_ranges)

_MUNG_EXTENSION_GLYPHNAMES = {
    "beam",
    "bracket",
    "augmentationDot",
    "slur",
    "stem",
    "tie",
}
_DOLORES_EXTENSION_GLYPHNAMES = {
    "ending",
}

_ALL_GLYPHNAMES = {k for k in _DATA_GLYPHNAMES.keys()}.union(
    _MUNG_EXTENSION_GLYPHNAMES,
    _DOLORES_EXTENSION_GLYPHNAMES,
)

_NAME2CODEPOINT = {k: v["codepoint"] for k, v in _DATA_GLYPHNAMES.items()}
_CODEPOINT2NAME = {v: k for k, v in _NAME2CODEPOINT.items()}


class Glyph(str):
    RE_HEX = re.compile(r"[0-9a-fA-F]{4}")

    def __new__(cls, *values: str):
        for val in values:
            if not isinstance(val, str):
                raise TypeError("Glyph must be a string")
            if val not in _ALL_GLYPHNAMES:
                raise ValueError(f"{val} is not a supported SMuFL glyph name")

        value = "+".join(values)
        return str.__new__(cls, value)

    @classmethod
    def from_codepoint(cls, src: str) -> "Glyph":
        """Construct a Glyph from a codepoint with form U+XXXX fed as a string."""
        return cls(_CODEPOINT2NAME[src])

    @classmethod
    def from_multiple_codepoints(cls, *codepoints: str):
        """Construct a Glyph from a list of codepoints."""
        return cls(*map(_CODEPOINT2NAME.get, codepoints))
