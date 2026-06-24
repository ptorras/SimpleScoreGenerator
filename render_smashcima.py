from pathlib import Path
from subprocess import CalledProcessError
from zipfile import ZipFile

import cv2
import smashcima as sc
from lxml import etree
from tqdm.auto import tqdm

# MuseScore.configure(Path(r"/Applications/MuseScore 4.app/Contents/MacOS/mscore"))
score_path = Path("/Users/ptorras/Documents/Datasets/ComrefMusicxml")
target_path = Path("/Users/ptorras/Documents/Datasets/SmashcimaComref")
target_path.mkdir(exist_ok=True)

for mxml_path in score_path.glob("*.mxl"):
    with ZipFile(mxml_path) as f_zip:
        file_list = [
            x for x in f_zip.namelist() if "META-INF/" not in x and x[-3:] == "xml"
        ]
        with (
            f_zip.open(file_list[0], "r") as xml_in,
            open(mxml_path.with_suffix(".musicxml"), "wb") as f_out,
        ):
            f_out.write(xml_in.read())

# job = [
#     {"in": str(mxml), "out": str(mxml.with_suffix(".musicxml"))}
#     for mxml in score_path.glob("*.mxl")
# ]

# try:
#     MuseScore.run_musescore_job(job)
# except CalledProcessError as e:
#     print(e.stderr)
#     print(e.stdout)
#     raise

# for score in tqdm([x["out"] for x in job]):

for score in tqdm(list(score_path.glob("*.musicxml"))):
    try:
        model = sc.orchestration.BaseHandwrittenModel()
        exporter = sc.exporting.SvgExporter(render_labeled_regions=True)

        scene = model(score)

        for ii, page in enumerate(scene.pages):
            bitmap = scene.render(page)
            cv2.imwrite(str(target_path / f"{score.stem}_page_{ii}.png"), bitmap)
            svg = exporter.export_string(page.view_box)
            with open(target_path / f"{score.stem}_page_{ii}.svg", "w") as f:
                f.write(svg)
    except Exception as exc:
        print(exc)
        continue
