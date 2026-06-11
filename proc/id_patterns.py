import re


RE_ACCENT = re.compile(
    r"^line\d+:(?:chord\.\w+|\w+)\.articulation\d+\.accent$"
)  # accent
RE_ACCIDENTAL = re.compile(r"^line\d+:(?:chord\.\w+|\w+)\.accidental\d*$")  # accidental
RE_BARLINE = re.compile(r"^line\d+:\w+\.(\w+\.)?barline_tok\d+$")  # barline
RE_BEAM = re.compile(r"^line\d+:beam\d+$")  # beam
RE_BRACKET = re.compile(r"^line\d+:(?:chord\.\w+|\w+).bracket$")  # bracket
RE_CAESURA = re.compile(
    r"^line\d+:(?:chord\.\w+|\w+)\.articulation\d+\.caesura$"
)  # caesura
RE_CLEF = re.compile(r"^line\d+:clef\d+(_\d+)?$")  # clef
RE_CODA = re.compile(r"^line\d+:clef\d+_\d+$")  # coda (not used)
RE_DOTS = re.compile(r"^line\d+:(?:chord\.\w+|\w+)\.dot\d+$")  # dots
RE_DYNAMICS = re.compile(r"^line\d+:clef\d+_\d+$")  # dynamics (not used)
# RE_ENDING = re.compile()  # ending (not used)
RE_FERMATA = re.compile(r"^line\d+:fermata\d+$")  # fermata
RE_FLAG = re.compile(r"^line\d+:(?:chord\.\w+|\w+)\.(?:stem\.)?flag$")  # flag
# RE_GLISSANDO = re.compile()  # glissando
# RE_MORDENT = re.compile()  # mordent
# RE_TURN = re.compile()  # turn
# RE_MEASURE_REPEAT = re.compile()  # measure_repeat
RE_NOTEHEAD = re.compile(r"^(?:chord\.\w+|\w+)\.notehead(?:\d+)?$")  # notehead
# RE_NUMBER = re.compile()  # number
# RE_OCTAVE_SHIFT = re.compile()  # octave_shift
RE_REST = re.compile(r"^line\d+:rest\d+$")  # rest
# RE_SCHLEIFER = re.compile()  # schleifer
# RE_SEGNO = re.compile()  # segno
# RE_SLUR = re.compile()  # slur
RE_STACCATO = re.compile(
    r"^line\d+:(?:chord\.\w+|\w+)\.articulation\d+\.staccato$"
)  # staccato
RE_STEM = re.compile(r"^line\d+:(?:chord\.\w+|\w+)\.stem$")  # stem
# RE_TENUTO = re.compile()  # tenuto
# RE_TIE = re.compile()  # tie
RE_TIMESIG = re.compile(r"^line\d+:time\d+_\d+(_\d+)?$")  # timesig
RE_TREMOLO_LINE = re.compile(
    r"^line\d+:(?:chord\.\w+|\w+)\.tremolo_single\.line\d+$"
)  # tremolo line
RE_BEAM_CHORD = re.compile(
    r"^line\d+:(?:chord\.\w+|\w+)\.tremolo_beam\.line\d+$"
)  # tremolo beam
# RE_TRILL = re.compile()  # trill
# RE_WEDGE = re.compile()  # wedge
# RE_REPEAT = re.compile()  # repeat
