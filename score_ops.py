import random
from copy import deepcopy

import music21 as m21


def create_test_part(score: m21.stream.Score) -> m21.stream.Score:
    score = m21.stream.Score(
        m21.stream.Part(score.parts[-1].getElementsByClass(m21.stream.Measure)[:4])
    )
    score = deepcopy(score)
    return score


def turn_measure_monophonic(measure: m21.stream.Measure) -> m21.stream.Measure:
    measure.flattenUnnecessaryVoices(inPlace=True)
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


def randomly_modify_pitches(
    music: m21.stream.Stream,
    probability: float = 0.3,
    dist: int = 7,
    in_place: bool = True,
) -> m21.stream.Stream:
    if not in_place:
        music = deepcopy(music)
    for note in music.recurse(classFilter=m21.note.Note):
        if random.random() <= probability:
            transpose_to = random.randint(-dist, dist)
            note.transpose(transpose_to, inPlace=True)
    return music


def randomly_convert_to_rest(
    music: m21.stream.Stream,
    probability: float = 0.1,
    in_place: bool = True,
) -> m21.stream.Stream:
    if not in_place:
        music = deepcopy(music)
    to_replace = []
    for note in music.recurse(classFilter=m21.note.Note):
        if random.random() <= probability:
            newrest = m21.note.Rest()
            newrest.duration = note.duration

            to_replace.append((note, newrest, note.activeSite))
    for note, newrest, active_site in to_replace:
        active_site.replace(note, newrest)

    return music


def randomly_modify_main_key(
    music,
    probability: float = 0.1,
    in_place: bool = True,
) -> m21.stream.Stream:
    if not in_place:
        music = deepcopy(music)
    if len(list(music.recurse(classFilter=m21.key.KeySignature))) == 0:
        if random.random() <= probability:
            music.insert(0, m21.key.KeySignature(random.randint(-7, 7)))
    else:
        for key in music.recurse(classFilter=m21.key.KeySignature):
            if random.random() <= probability:
                num = random.randint(-7, 7)
                key.sharps = num
    return music


def randomly_insert_key_change(
    music: m21.stream.Stream,
    probability: float = 0.1,
    in_place: bool = True,
) -> m21.stream.Stream:
    if not in_place:
        music = deepcopy(music)
    notes = list(music.recurse(classFilter=m21.note.Note))
    if len(notes) > 3 and random.random() <= probability:
        position = random.randint(0, len(notes) - 1)
        sharps = random.randint(-7, 7)
        notes[position].activeSite.insert(
            notes[position].offset, m21.key.KeySignature(sharps)
        )

    return music


def rebuild_beams(music: m21.stream.Stream) -> m21.stream.Stream:
    for part in music.getElementsByClass(m21.stream.Part):
        if part.streamStatus.haveBeamsBeenMade():
            for elm in part.recurse().notes:
                elm.beams = None
        part.makeBeams(inPlace=True)
    return music
