from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Iterator, List

import lxml.etree as etree
import numpy as np
from lxml.etree import _Element as Element

from .smufl import Glyph

_LOGGER = logging.getLogger(__name__)

NAMESPACES = {
    "svg": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
    "mei": "http://www.music-encoding.org/ns/mei",
}


@dataclass
class Point:
    """Represents a point in a bounding box."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.x},{self.y}"

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y

    def dist(self, other: "Point") -> float:
        return sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)


@dataclass
class RepeatDot(Point):
    character: str

    def to_svg(self) -> Element:
        return etree.Element(
            "use",
            {
                f"{{{NAMESPACES['xlink']}}}href": str(self.character),
                "x": str(self.x),
                "y": str(self.y),
                "height": "720px",
                "width": "720px",
            },
        )


class RepeatType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class RepeatDots:
    def __init__(
        self,
        point_top: RepeatDot,
        point_bot: RepeatDot,
        direction: RepeatType,
    ) -> None:
        self.point_top = point_top
        self.point_bot = point_bot
        self.direction = direction

    def to_svg(self, ident: str) -> Element:
        output = etree.Element(
            f"{{{NAMESPACES['svg']}}}g",
            {"id": ident, "class": "repeat"},
            nsmap=NAMESPACES,
        )

        output.append(self.point_top.to_svg())
        output.append(self.point_bot.to_svg())

        return output


@dataclass
class Rectangle:
    tl: Point
    tr: Point
    br: Point
    bl: Point

    def __iter__(self) -> Iterator[Point]:
        yield from [self.tl, self.tr, self.br, self.bl]


@dataclass
class SvgLine:
    origin: Point
    dest: Point
    weight: int

    def to_svg(self, ident: str) -> Element:
        output = etree.Element(
            f"{{{NAMESPACES['svg']}}}path",
            {
                "id": ident,
                "d": f"M{self.origin.x} {self.origin.y} L{self.dest.x} {self.dest.y}",
                "stroke": "currentColor",
                "stroke-width": str(self.weight),
                "class": "barline_tok",
            },
            nsmap=NAMESPACES,
        )
        return output


class SVGProcessor:
    """Processes an SVG file to incorporate the required identifiers."""

    RE_MOVETO_COMMAND = re.compile(r"M(\d+)[, ](\d+)")
    RE_LINETO_COMMAND = re.compile(r"L(\d+)[, ](\d+)")
    RE_BEAM_ID = re.compile(r"beam(\d+)")
    RE_XPATH_GLYPH_ID = re.compile(r"#([A-Fa-f0-9]+)-\w+")

    SMUFL_CREATE_FROM_NAME = {
        "beam": "beam",
        "bracket": "bracket",
        "dot": "augmentationDot",
        "ending": "ending",
        "slur": "slur",
        "stem": "stem",
        "tie": "tie",
    }

    SVG2CLASS = {
        "accid": "accidental",
        "artic": "artic",
        "barline_tok": "barline_tok",
        "beam": "beam",
        "bracketSpan": "bracket",
        "bTrem_line": "beam",
        "caesura": "caesura",
        "clef": "clef",
        "coda": "coda",
        "dir coda": "coda",
        # "dynam": "dyn",
        "ending": "ending",
        "ending systemMilestone": "ending",
        "fermata": "fermata",
        "flag": "flag",
        "fTrem_line": "stem",
        # "glissando": "glissando",
        # "hairpin": "wedge",
        "keyAccid": "accidental",
        "meterSig": "timesig",
        "mordent": "mordent",
        "mRest": "rest",
        "mRpt": "measure_repeat",
        "multiRest": "rest",
        "notehead": "notehead",
        "octave": "octave_shift",
        "repeat": "repeat",
        "rest": "rest",
        "segno": "segno",
        "dir segno": "segno",
        "single_dot": "dot",
        "slur": "slur",
        "stem": "stem",
        "tie": "tie",
        "trill": "trill",
        "tupletBracket": "bracket",
        "tupletNum": "number",
        "turn": "turn",
    }

    CATEGORY_PATHS = " or ".join(
        [f"@class='{category}'" for category in SVG2CLASS.keys()]
    )
    XPATH = f".//*[{CATEGORY_PATHS}]"

    def init(self) -> None:
        self.beam_id = 1

    def process(self, svg_file: Path) -> None:
        """Open an SVG file and perform changes required by the DoLoReS project.

        Parameters
        ----------
        svg_file : Path
            Path to the SVG file to modify.
        """
        self.beam_id = 1

        tree = etree.parse(svg_file)
        root = tree.getroot()

        self._remove_unnecessary_svg(root)
        self._rebuild_svg_beams(root)
        self._rebuild_svg_barlines(root)

        # Performed AFTER changing beams - careful!
        self._identify_svg_attributes(root)
        # self._identify_svg_timesigs(root)
        self._identify_svg_dots(root)
        self._identify_svg_noteheads(root)
        self._identify_svg_tremolos(root)
        self._identify_svg_flags(root)
        self._identify_svg_tuplet_num(root)
        self._identify_svg_tuplet_bracket(root)
        self._identify_svg_mrep(root)
        self._identify_svg_ending(root)

        etree.indent(tree, "    ")
        tree.write(svg_file)

    def extract_objects(self, svg_file: Path) -> list[tuple[str, str, Glyph]]:
        """Produce a list of all objects to annotate within an SVG with metadata

        Parameters
        ----------
        svg_file : Path
            Path to the SVG file to introspect.

        Returns
        -------
        list[tuple[str, str, Glyph]]
            A list of (class_name, mxml_id, smufl_glyph) tuples as found in the SVG.

        Raises
        ------
        ValueError
            If a null-valued mxml identifier is found.
        ValueError
            If an articulation that is not covered by the class repertoire is found.
        ValueError
            If a mordent that is not covered by the class repertoire is found.
        """
        tree = etree.parse(svg_file)
        root = tree.getroot()

        elms = []
        for elm in root.xpath(self.XPATH):
            mxml_id = elm.get("id")
            svg_class = elm.get("class")
            if mxml_id is None:
                warnings.warn("Skipping object because no available MXML id")
                continue
                # raise ValueError("Null value in mxml identifier")
            try:
                class_name = self._extract_class_name(svg_class, mxml_id)
                smufl_id = self._extract_smufl_id(elm, class_name)
                elms.append((class_name, mxml_id, smufl_id))
            except Exception:
                continue
        return elms

    def _extract_class_name(
        self,
        svg_class: str,
        mxml_id: str,
    ) -> str:
        class_name = self.SVG2CLASS[svg_class]
        if class_name == "artic":
            if "caesura" in mxml_id:
                class_name = "caesura"
            elif "staccatissimo" in mxml_id:
                class_name = "staccato"
            elif "staccato" in mxml_id:
                class_name = "staccato"
            elif "accent" in mxml_id:
                class_name = "accent"
            elif "tenuto" in mxml_id:
                class_name = "tenuto"
            elif "spiccato" in mxml_id:
                class_name = "staccato"
            else:
                class_name = "other"
        if class_name == "mordent":
            if "mordent" in mxml_id:
                class_name = "mordent"
            elif "schleifer" in mxml_id:
                class_name = "schleifer"
            else:
                raise ValueError(f"Mordent {mxml_id} not addressed")
        return class_name

    def _barline_smufl(self, barline: Element) -> Glyph:
        """Find the SMuFL class of a barline.

        In this implementation we rely on the fact that Verovio produces consistent
        stroke widths and objects depending on the type of Barline. If this changes
        we would probably need to adjust this algorithm (although we are sticking to
        a specific version of Verovio precisely for this reason).

        Arguments
        ---------
        barline: Element
            The barline_tok class object to extract the SMuFL class from.

        Returns
        -------
        Glyph
            The corresponding SMuFL Glyph.
        """
        obj_class = barline.get("class")
        if obj_class is None or obj_class != "barline_tok":
            raise ValueError(f"Trying to extract barline SMuFL from {obj_class} object")

        stroke_width = barline.get("stroke-width")
        if stroke_width is None:
            raise ValueError("Stroke width not set for barline: cannot determine type")

        # INFO We can differentiate between dotted and dashed barlines with this method
        # but since we do not support them anyway yet we can just get away with this.

        stroke_width = int(stroke_width)
        if stroke_width < 50:
            return Glyph("barlineSingle")
        else:
            return Glyph("barlineHeavy")

    def _extract_smufl_id(self, elm: Element, class_name: str) -> Glyph:
        if class_name in {
            "accent",
            "accidental",
            "caesura",
            "clef",
            "fermata",
            "flag",
            "measure_repeat",
            "mordent",
            "notehead",
            "number",
            "octave_shift",
            "repeat",
            "rest",
            "schleifer",
            "staccato",
            "tenuto",
            # "timesig",
            "trill",
            "turn",
        }:
            # NOTE Accents need to be differentiated as being above or below the note.
            # Fortunately, Verovio adds the specific glyph via the "use" child.
            # (same is true for many other symbols actually!)

            # NOTE The way the dataset is annotated we cannot distinguish among the
            # different symbols that comprise a compound number time signature. It
            # would probably be doable computationally to split each timesig annotation
            # into two and add each individual glyph as a symbol, but we would have to
            # change quite a bit of stuff. Regardless, it is possible to extract this
            # info with a bit of patience. Common time signatures and so on are just
            # a symbol that can be extracted well using this technique, so it's all
            # good.

            # NOTE Octave shifts contain plenty of elements. Nevertheless, the key SMuFL
            # symbol is the ottava bassa or ottava alta indicator. This is the first
            # element with an xlink directive. Thus, extracting it is trivial.
            try:
                smufl_object = self._extract_smufl_from_child_use_node(elm)
            except IndexError:
                # INFO For some reason, some noteheads are drawn via polygons. Try to
                # construct them with an emergency function here.
                smufl_object = self._backup_extract_smufl_from_element(elm)
            return smufl_object
        if class_name == "timesig":
            return self._extract_smufl_from_multiple_child_use_node(elm)
        if class_name in self.SMUFL_CREATE_FROM_NAME:
            # NOTE Beams are tricky. SMuFL does not define a generic construct for beams
            # at all; they only offer beams for notes that are to be placed next to text.
            # In MUSCIMA++ they use their own primitives for these such objects that
            # SMuFL does not cover directly. We believe it is best to try to stay
            # compatible with existing schemas and use the same philosophy, using the
            # same string-based names as proposed in MuNG -- in this case, "stem".
            # Other elements that we are supporting in this "roundabout" manner are
            # slurs, endings and similar. Extension elements are documented in the SMuFL
            # file where appropriate.

            # FIXME In the future, some of the extension planes of Unicode could be
            # used to give these extensions a codepoint, so that they can be encoded
            # easily instead of having to use text.

            # FIXME Currently we do not have any "exotic" stems in the database, but
            # should we encounter variants such as "sul ponticello", we would have to
            # cover them here.
            return Glyph(self.SMUFL_CREATE_FROM_NAME[class_name])
        if class_name == "barline_tok":
            return self._barline_smufl(elm)
        if class_name == "dyn":
            # FIXME The dataset does not annotate these. If somebody wants to support
            # them in the future, they can do so via this point of the code.
            raise NotImplementedError()
        if class_name == "glissando":
            # FIXME There do not seem to be any in the database. Same as before.
            raise NotImplementedError()
        if class_name in {"segno", "coda"}:
            return self._extract_smufl_from_segno(elm)
        if class_name == "wedge":
            # FIXME For now there do not seem to be wedges being annotated in the
            # dataset.
            raise NotImplementedError()
            return Glyph.from_codepoint("")
        raise ValueError(
            f"No SMuFL ID can be produced for object {elm.tag} class: {class_name}"
        )

    def _backup_extract_smufl_from_element(self, elm: Element) -> Glyph:
        svg_class = elm.get("class")

        if svg_class is None:
            raise ValueError("No class property for object")

        if svg_class == "notehead":
            if (
                len(elm) == 3
                and elm[0].tag == r"{http://www.w3.org/2000/svg}polygon"
                and elm[1].tag == r"{http://www.w3.org/2000/svg}polygon"
                and elm[2].tag == r"{http://www.w3.org/2000/svg}rect"
            ):
                return Glyph("noteheadDoubleWholeSquare")

        raise ValueError(f"No backup operation for element {elm}")

    def _extract_smufl_from_segno(self, elm: Element) -> Glyph:
        """Get SMuFL code from a segno construct in the SVG.

        Segnos are strange in that the glyph code is set directly as text without the
        hex code in the SVG. We have to do a bit of digging in order to extract this
        info.

        Parameters
        ----------
        elm: Element
            The SVG element to extract this info from.

        Returns
        -------
        Glyph
            Glyph with the unicode rune pointing to this character.
        """
        rune = elm.xpath(
            "./svg:text/svg:tspan/svg:tspan/svg:tspan", namespaces=NAMESPACES
        )[0].text
        return Glyph.from_codepoint(f"U+{hex(ord(rune))[2:].upper()}")

    def _extract_smufl_from_child_use_node(self, elm: Element) -> Glyph:
        """Get the SMuFL id from a child "use" element.

        Verovio inserts preset SMuFL glyphs such as clefs and noteheads by using xlink
        and use. The identifier of the reference object happens to contain the SMuFL
        codepoint of the element in question as its first characters, so we can obtain
        the SMuFL ID of an object very easily with a regular expression.

        This function extracts this information from the *first* of the "use" child
        contained within the node passed as argument. The rest are ignored.

        Parameters
        ----------
        elm: Element
            The lxml.Element object that contains the use child.

        Returns
        -------
        Glyph
            The glyph extracted from the codepoint in the identifier.

        Raises
        ------
        ValueError
            If the node's child element contains a malformed id.
        IndexError
            If the node does not contain a use child that can be used to construct
            a SMuFL identifier.
        """
        ident = elm.xpath("./svg:use/@xlink:href", namespaces=NAMESPACES)[0]
        match = self.RE_XPATH_GLYPH_ID.match(ident)
        if match is None:
            raise ValueError(
                "The selected child element has no SMuFL id or it is not properly "
                "formatted"
            )
        return Glyph.from_codepoint(f"U+{match.group(1)}")

    def _extract_smufl_from_multiple_child_use_node(self, elm: Element) -> Glyph:
        """Get the SMuFL id from all child "use" elements.

        Verovio inserts preset SMuFL glyphs such as clefs and noteheads by using xlink
        and use. The identifier of the reference object happens to contain the SMuFL
        codepoint of the element in question as its first characters, so we can obtain
        the SMuFL ID of an object very easily with a regular expression.

        Parameters
        ----------
        elm: Element
            The lxml.Element object that contains the use children.

        Returns
        -------
        Glyph
            The combined glyph extracted from the codepoint in the identifier.

        Raises
        ------
        ValueError
            If the node's child element contains a malformed id.
        IndexError
            If the node does not contain a use child that can be used to construct
            a SMuFL identifier.
        """
        idents = elm.xpath("./svg:use/@xlink:href", namespaces=NAMESPACES)
        codepoints = []
        for ident in idents:
            match = self.RE_XPATH_GLYPH_ID.match(ident)
            if match is None:
                raise ValueError(
                    "The selected child element has no SMuFL id or it is not properly "
                    "formatted"
                )
            codepoints.append(f"U+{match.group(1)}")
        return Glyph.from_multiple_codepoints(*codepoints)

    def _remove_unnecessary_svg(self, root: Element) -> None:
        """Remove empty SVG group elements and other minor annoyances.

        Parameters
        ----------
        root : Element
            Root element of the score in SVG format.
        """
        group_elements = root.findall(".//svg:g", namespaces=NAMESPACES)
        for child in group_elements:
            if len(child) == 0:
                child.getparent().remove(child)
                _LOGGER.debug(f"Removing subtree: {etree.tostring(child)}")

    def _rebuild_svg_beams(self, root: Element) -> None:
        """Change Verovio beam fragments into continuous beams that can be identified.

        Verovio segments beams into segments. If the initial geometry of the beam is
        something like:

        +--+--+--+
        +--+--|  |
        |  |  |  |
        O  O  O  O

        This will be converted into 3 different segments. The top beam will always be
        one singular object, whereas beams below will be segmented into fragments
        spanning the full width of the space between stems. Thus:

        +11+11+11+
        +22+33|  |
        |  |  |  |
        O  O  O  O

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        beam_nodes = root.findall(".//svg:g[@class='beam']", namespaces=NAMESPACES)

        for beam_node in beam_nodes:
            beam_id = beam_node.get("id", "")
            beam_id_match = self.RE_BEAM_ID.match(beam_id)

            if beam_id_match is None:
                raise ValueError("Invalid beam id formatting")
            # id_index = int(beam_id_match.group(1))

            new_beams = []

            beam_fragments = beam_node.findall("./svg:polygon", namespaces=NAMESPACES)
            if len(beam_fragments) == 0:
                raise ValueError("Beam without drawn polygons")

            prev_frag = self._get_beam_rectangle(beam_fragments[0])

            for curr_frag in beam_fragments[1:]:
                curr_frag = self._get_beam_rectangle(curr_frag)

                if prev_frag.tr == curr_frag.tl and prev_frag.br == curr_frag.bl:
                    prev_frag = Rectangle(
                        prev_frag.tl, curr_frag.tr, curr_frag.br, prev_frag.bl
                    )
                else:
                    new_beams.append(prev_frag)
                    prev_frag = curr_frag

            new_beams.append(prev_frag)

            for frag in beam_fragments:
                beam_node.remove(frag)

            for new_beam in reversed(new_beams):
                beam_node.insert(
                    0,
                    etree.Element(
                        "polygon",
                        attrib={
                            "points": " ".join(map(str, new_beam)),
                            "id": f"beam{self.beam_id}",
                            "class": "beam",
                        },
                        nsmap=NAMESPACES,
                    ),
                )
                self.beam_id += 1
            beam_node.set("id", beam_node.get("id", "") + "_parent")
            beam_node.set("class", beam_node.get("class", "") + "_parent")

    def _get_beam_rectangle(self, et_poly: Element) -> Rectangle:
        points = et_poly.get("points")

        if points is None:
            raise ValueError("Beam polygon element has no points attribute")

        points = points.split(" ")

        if len(points) != 4:
            raise ValueError("More than 4 points present in rectangle polygon")

        tl, tr, br, bl = map(lambda x: Point(*map(int, x.split(","))), points)

        return Rectangle(tl, tr, br, bl)

    def _rebuild_svg_barlines(self, root: Element) -> None:
        barlines = root.findall(".//svg:g[@class='barLine']", namespaces=NAMESPACES)
        for barline in barlines:
            self._edit_barline_elements(barline)

    def _edit_barline_elements(self, barline_node: Element) -> None:
        barline_id = barline_node.get("id")

        # Make sure the barline element is not empty
        if len(barline_node) == 0:
            parent = barline_node.getparent()
            if parent is not None:
                parent.remove(barline_node)
            return
        # FIXME: This does not work with dotted barlines!
        segments = barline_node.findall("./svg:path", namespaces=NAMESPACES)
        segments = list(map(self._parse_segment, segments))
        segments = self._combine_segments(segments)

        dots = barline_node.findall("./svg:use", namespaces=NAMESPACES)
        dots = list(map(self._parse_repeat_dot, dots))
        dots = self._combine_repeat_dots(dots, segments[0].origin.x)

        for node in barline_node:
            barline_node.remove(node)

        for ii, segment in enumerate(segments, 1):
            barline_node.append(segment.to_svg(f"{barline_id}.barline_tok{ii}"))

        for ii, dot in enumerate(
            [x for x in dots if x.direction == RepeatType.FORWARD], 1
        ):
            barline_node.append(dot.to_svg(f"{barline_id}.repeat_forward{ii}"))

        for ii, dot in enumerate(
            [x for x in dots if x.direction == RepeatType.BACKWARD], 1
        ):
            barline_node.append(dot.to_svg(f"{barline_id}.repeat_backward{ii}"))

    def _parse_segment(self, path_element: Element) -> SvgLine:
        draw_cmd = path_element.get("d")

        if draw_cmd is None:
            raise ValueError("No draw command in Barline segment")

        weight = path_element.get("stroke-width")

        if weight is None:
            raise ValueError("No stroke-width command in Barline segment")

        weight = int(weight)

        move_cmd = self.RE_MOVETO_COMMAND.search(draw_cmd)
        line_cmd = self.RE_LINETO_COMMAND.search(draw_cmd)

        if move_cmd is None or line_cmd is None:
            raise ValueError("Malformed draw command in Barline segment")

        origin_x, origin_y = map(int, move_cmd.groups())
        target_x, target_y = map(int, line_cmd.groups())

        return SvgLine(Point(origin_x, origin_y), Point(target_x, target_y), weight)

    def _combine_segments(self, lines: List[SvgLine]) -> List[SvgLine]:
        sorted_lines = list(sorted(lines, key=lambda x: (x.origin.x, x.origin.y)))
        output_lines = []
        base_segment = sorted_lines[0]

        for comp_segment in sorted_lines[1:]:
            if (
                base_segment.dest.y == comp_segment.origin.y
                and base_segment.origin.x == comp_segment.origin.x
                and base_segment.weight == comp_segment.weight
            ):
                base_segment.dest = comp_segment.dest
            else:
                output_lines.append(base_segment)
                base_segment = comp_segment
        output_lines.append(base_segment)
        return output_lines

    def _parse_repeat_dot(self, dot_element: Element) -> RepeatDot:
        character = dot_element.get(f"{{{NAMESPACES['xlink']}}}href")
        xcoord = dot_element.get("x")
        ycoord = dot_element.get("y")

        if character is None:
            raise ValueError("No string value in repeat dot element")

        if xcoord is None or ycoord is None:
            raise ValueError("Missing coordinate in repeat dot element")

        return RepeatDot(int(xcoord), int(ycoord), character)

    def _combine_repeat_dots(
        self, points: List[RepeatDot], reference_x: int
    ) -> List[RepeatDots]:
        sorted_dots = list(sorted(points, key=lambda x: (x.x, x.y)))
        output_dots = []
        for dot1, dot2 in zip(sorted_dots[::2], sorted_dots[1::2]):
            if dot1.x == dot2.x:
                output_dots.append(
                    RepeatDots(
                        dot1,
                        dot2,
                        (
                            RepeatType.BACKWARD
                            if dot1.x < reference_x
                            else RepeatType.FORWARD
                        ),
                    )
                )
        return output_dots

    def _identify_svg_attributes(self, root: Element) -> None:
        """Give an identifier to time signature elements.

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        for symtype in ["meterSig", "keySig", "clef"]:
            containers = root.findall(
                f".//svg:g[@class='{symtype}']", namespaces=NAMESPACES
            )
            memory = {}
            for container in containers:
                container_id = container.get("id")

                if container_id in memory:
                    container.set("id", f"{container_id}_{memory[container_id]}")
                    memory[container_id] += 1
                else:
                    container.set("id", f"{container_id}_1")
                    memory[container_id] = 2

                if symtype == "keySig":
                    for ii, accid in enumerate(
                        container.findall(
                            ".//svg:g[@class='keyAccid']", namespaces=NAMESPACES
                        ),
                        1,
                    ):
                        accid.set("id", f"{container.get('id')}.accidental{ii}")

    def _identify_svg_timesigs(self, root: Element) -> None:
        """Give an identifier to time signature elements.

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        time_containers = root.findall(
            ".//svg:g[@class='meterSig']", namespaces=NAMESPACES
        )
        memory = {}
        for container in time_containers:
            container_id = container.get("id")

            if container_id in memory:
                container.set("id", f"{container_id}_{memory[container_id]}")
                memory[container_id] += 1
            else:
                container.set("id", f"{container_id}_1")
                memory[container_id] = 2

    def _identify_svg_dots(self, root: Element) -> None:
        """Give an identifier to dots.

        Parameters
        ----------
        root : Element
            Root SVG score element.

        """
        # Find dots elements within other elements
        dot_containers = root.xpath(".//*[svg:g[@class='dots']]", namespaces=NAMESPACES)
        for container in dot_containers:
            container_id = container.get("id", None)

            if container_id is None:
                raise ValueError("Container has null id")

            container_class = container.get("class", None)

            if container_class is None:
                raise ValueError("Container has no known class")

            dots_element = container.find(
                "./svg:g[@class='dots']", namespaces=NAMESPACES
            )
            if dots_element is None:
                raise ValueError("For some reason there are no dots on a dot query")
            dots_element.set("id", f"{container_id}.dots_parent")

            # If the object is under a note, it is easy to process because we only need
            # to set the id of the parent object
            if container_class in {"note", "rest"}:
                for ii, ellipse in enumerate(dots_element, 1):
                    ellipse.set("id", container_id + f".dot{ii}")
                    ellipse.set("class", "single_dot")
                continue

            # Otherwise, we have to find all noteheads within the container and assign
            # each dot to the closest note.
            dot_coords = []
            for dot in dots_element:
                # Should be an ellipse
                dot.set("class", "single_dot")

                x_dot = dot.get("cx")
                y_dot = dot.get("cy")

                if x_dot is None or y_dot is None:
                    raise ValueError(
                        "Dot ellipse has no center. Can't identify SVG dots."
                    )

                x_dot = int(x_dot)
                y_dot = int(y_dot)

                dot_coords.append(Point(x_dot, y_dot))

            noteheads = container.xpath(
                ".//svg:g[@class='note']/svg:g[@class='notehead']/svg:use",
                namespaces=NAMESPACES,
            )
            notehead_coords = []

            for notehead in noteheads:
                # Should be an ellipse

                x_notehead = notehead.get("x")
                y_notehead = notehead.get("y")

                assert (
                    x_notehead is not None and y_notehead is not None
                ), "Notehead has no center"

                x_notehead = int(x_notehead)
                y_notehead = int(y_notehead)

                notehead_coords.append(Point(x_notehead, y_notehead))

            notehead_matrix = np.array(
                list(map(lambda x: x.as_tuple(), notehead_coords))
            )
            dot_matrix = np.array(list(map(lambda x: x.as_tuple(), dot_coords)))

            # Use only y coordinates
            dist_matrix = (
                dot_matrix[:, np.newaxis, 1] - notehead_matrix[np.newaxis, :, 1]
            )
            dist_matrix = np.abs(dist_matrix)

            dot_indices = dist_matrix.argmin(1)

            repeat_dots = {ii: 1 for ii in range(len(notehead_matrix))}

            for dot_ind, note_ind in enumerate(dot_indices):
                note_ob = noteheads[note_ind].getparent().getparent()
                note_id = note_ob.get("id", None)

                dots_element[dot_ind].set("id", f"{note_id}.dot{repeat_dots[note_ind]}")
                repeat_dots[note_ind] += 1

    def _identify_svg_tremolos(self, root: Element) -> None:
        """Give an identifier to tremolos.

        Verovio will provide the identifier of the first note of the tremolo group or
        tremolo stem. This function propagates the id alongside an index to the various
        line elements that form the tremolo, as well as giving them a "tremolo_line"
        class.

        Parameters
        ----------
        root : Element
            Root SVG score element.

        """
        ftrem_objects = root.findall(".//svg:g[@class='fTrem']", namespaces=NAMESPACES)
        for ftrem in ftrem_objects:
            ident = ftrem.get("id")
            for ii, line in enumerate(
                ftrem.findall("./svg:polygon", namespaces=NAMESPACES), 1
            ):
                line.set("id", f"{ident}.line{ii}")
                line.set("class", "fTrem_line")

        btrem_objects = root.findall(".//svg:g[@class='bTrem']", namespaces=NAMESPACES)
        for btrem in btrem_objects:
            ident = btrem.get("id")
            for ii, line in enumerate(
                btrem.findall("./svg:use", namespaces=NAMESPACES), 1
            ):
                line.set("id", f"{ident}.line{ii}")
                line.set("class", "bTrem_line")

    def _identify_svg_noteheads(self, root: Element) -> None:
        """Provide an identifier to notehead objects in the SVG.

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        note_nodes = root.findall(".//svg:g[@class='note']", namespaces=NAMESPACES)
        for note_node in note_nodes:
            notehead_node = note_node.find(
                "./svg:g[@class='notehead']", namespaces=NAMESPACES
            )
            if notehead_node is not None:
                notehead_node.set("id", f"{note_node.get('id')}.notehead")

    def _identify_svg_mrep(self, root: Element) -> None:
        """Provide an identifier to mrep objects in the SVG.

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        measure_nodes = root.xpath(".//svg:g[@class='measure']", namespaces=NAMESPACES)
        for measure_node in measure_nodes:
            measure_repeat = measure_node.find(
                ".//svg:g[@class='mRpt']", namespaces=NAMESPACES
            )
            if measure_repeat is not None:
                measure_repeat.set("id", f"{measure_node.get('id')}.measure_repeat")

    def _identify_svg_ending(self, root: Element) -> None:
        """Provide an identifier to ending objects in the SVG.

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        ending_nodes = root.xpath(
            ".//svg:g[@class='ending systemMilestone']", namespaces=NAMESPACES
        )
        for ending_node in ending_nodes:
            bracket = ending_node.find(
                "./svg:g[@class='voltaBracket']", namespaces=NAMESPACES
            )
            if bracket is not None:
                bracket.set("id", f"{ending_node.get('id')}.bracket")

    def _identify_svg_flags(self, root: Element) -> None:
        """Provide an identifier to flag objects in the SVG.

        Parameters
        ----------
        root : Element
            Root SVG score element.
        """
        stem_nodes = root.findall(".//svg:g[@class='stem']", namespaces=NAMESPACES)

        for stem_node in stem_nodes:
            flag_node = stem_node.find("./svg:g[@class='flag']", namespaces=NAMESPACES)
            if flag_node is not None:
                flag_node.set("id", f"{stem_node.get('id')}.flag")

    def _identify_svg_tuplet_num(self, root: Element) -> None:
        """Provide an identifier to tuplet number objects in the SVG.

        Parameters
        ----------
        root : Element
            Root SVG score element.

        """
        tuplet_nodes = root.findall(".//svg:g[@class='tuplet']", namespaces=NAMESPACES)
        for tuplet_node in tuplet_nodes:
            number_node = tuplet_node.find(
                "./svg:g[@class='tupletNum']", namespaces=NAMESPACES
            )
            if number_node is not None:
                number_node.set("id", f"{tuplet_node.get('id')}.number")

    def _identify_svg_tuplet_bracket(self, root: Element) -> None:
        """Provide an identifier to tuplet bracket objects in the SVG.

        Parameters
        ----------
        root : Element
            Root SVG score element.

        """
        tuplet_nodes = root.findall(".//svg:g[@class='tuplet']", namespaces=NAMESPACES)
        for tuplet_node in tuplet_nodes:
            number_node = tuplet_node.find(
                "./svg:g[@class='tupletBracket']", namespaces=NAMESPACES
            )
            if number_node is not None:
                number_node.set("id", f"{tuplet_node.get('id')}.bracket")
