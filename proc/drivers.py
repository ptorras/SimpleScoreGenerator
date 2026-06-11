"""Simple singleton interface to engraving software."""

#  Copyright (c) Pau Torras 2025.
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
import logging
import uuid
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class Verovio:
    _VEROVIO_PATH: Optional[Path] = None

    @classmethod
    def configure(cls, verovio_path: Path) -> None:
        cls._VEROVIO_PATH = verovio_path

    @classmethod
    def run(cls, mxml_file: Path, svg_file: Path) -> None:
        if cls._VEROVIO_PATH is None:
            raise RuntimeError("You must run Verovio.configure() first!")
        cmd = run(
            args=[
                cls._VEROVIO_PATH.as_posix(),
                # "-a",
                "--adjust-page-height",
                "--adjust-page-width",
                "--breaks",
                "none",
                "--page-margin-bottom",
                "50",
                "--page-margin-left",
                "50",
                "--page-margin-right",
                "50",
                "--page-margin-top",
                "50",
                "--condense-first-page",
                "--header",
                "none",
                str(mxml_file),
                "-o",
                str(svg_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        _LOGGER.debug("Output for Verovio: " + cmd.stderr)
        if cmd.returncode != 0:
            raise ValueError(f"Return code for Verovio was not zero: {cmd.stderr}")


class MuseScore:
    _MUSESCORE_PATH: Optional[Path] = None
    MUSESCORE_VERSION: Optional[str] = None

    @classmethod
    def configure(cls, mscore_path: Path) -> None:
        cls._MUSESCORE_PATH = mscore_path
        cls.MUSESCORE_VERSION = cls._version()

    @classmethod
    def _version(cls) -> str:
        if cls._MUSESCORE_PATH is not None:
            cmd = run(
                args=[
                    cls._MUSESCORE_PATH.as_posix(),
                    "--version",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return cmd.stdout.split(" ")[-1].strip()
        else:
            raise ValueError("MuseScore path is None!")

    @classmethod
    def run_musescore_job(cls, job: List[Dict[str, str]]) -> None:
        if cls._MUSESCORE_PATH is None:
            raise RuntimeError("You must run MuseScore.configure() first!")
        job_path = Path(f"{uuid.uuid4()}.json")
        with open(job_path, "w") as f_job:
            json.dump(job, f_job, indent=4)
        try:
            run(
                args=[
                    cls._MUSESCORE_PATH.as_posix(),
                    "-j",
                    job_path,
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=cls._MUSESCORE_PATH.parent.as_posix(),
            )
        except CalledProcessError as e:
            raise e
        finally:
            job_path.unlink()

    @classmethod
    def run_musescore_single(cls, in_file: Path, out_file: Path) -> None:
        if cls._MUSESCORE_PATH is None:
            raise RuntimeError("You must run MuseScore.configure() first!")
        cmd = run(
            args=[
                cls._MUSESCORE_PATH.as_posix(),
                str(in_file),
                "-o",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=cls._MUSESCORE_PATH.parent.as_posix(),
        )

        if cmd.returncode != 0:
            raise ValueError(f"Return code for MuseScore was not zero: {cmd.stderr}")


class Verovio:
    _VEROVIO_PATH: Path | None = None

    @classmethod
    def configure(cls, verovio_path: Path) -> None:
        cls._VEROVIO_PATH = verovio_path

    @classmethod
    def run(cls, mxml_file: Path, svg_file: Path) -> None:
        if cls._VEROVIO_PATH is None:
            raise ValueError("You must run Verovio.configure() first!")
        cmd = run(
            args=[
                cls._VEROVIO_PATH.as_posix(),
                # "-a",
                "--adjust-page-height",
                "--adjust-page-width",
                "--breaks",
                "none",
                "--page-margin-bottom",
                "50",
                "--page-margin-left",
                "50",
                "--page-margin-right",
                "50",
                "--page-margin-top",
                "50",
                "--condense-first-page",
                "--header",
                "none",
                str(mxml_file),
                "-o",
                str(svg_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if cmd.returncode != 0:
            raise ValueError(f"Return code for Verovio was not zero: {cmd.stderr}")


class Inkscape:
    _INKSCAPE_PATH: Path | None = None

    @classmethod
    def configure(cls, inkscape_path: Path) -> None:
        cls._INKSCAPE_PATH = inkscape_path

    @classmethod
    def run(cls, svg_file: Path, png_file: Path) -> None:
        if cls._INKSCAPE_PATH is None:
            raise RuntimeError("You must run Inkscape.configure() first!")
        cmd = run(
            args=[
                cls._INKSCAPE_PATH.as_posix(),
                str(svg_file),
                "-o",
                str(png_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if cmd.returncode != 0:
            raise ValueError(f"Return code for Inkscape was not zero: {cmd.stderr}")

    @classmethod
    def run_bboxes(cls, svg_file: Path) -> dict[str, dict[str, int]]:
        if cls._INKSCAPE_PATH is None:
            raise RuntimeError("You must run Inkscape.configure() first!")
        cmd = run(
            args=[
                cls._INKSCAPE_PATH.as_posix(),
                "--query-all",
                str(svg_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out_elements = {}
        if cmd.stdout is not None:
            for line in cmd.stdout.strip().split("\n"):
                if len(line) == 0:
                    continue
                svg_id, x, y, w, h = line.split(",")
                out_elements[svg_id] = {
                    "x": int(float(x)),
                    "y": int(float(y)),
                    "w": int(float(w)),
                    "h": int(float(h)),
                }
        return out_elements
