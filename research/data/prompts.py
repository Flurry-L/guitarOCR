"""Visible-information prompts used to produce OCR training samples."""

import re

HEADER_PROMPT = (
    "Read only visible song title, artist or author credit (including composer, arranger or studio), "
    "and tuning label. Preserve the original language and spelling. Return one JSON object with "
    "title, artist, tuning_name; use null for absent fields. Tempo, playing techniques and chord names "
    "are not titles or author credits. Instrument and voice names are not song titles. "
    "An instrument's transposition designation is not a tuning label."
)
TEMPO_PROMPT = (
    "Read the printed quarter-note tempo. Return one JSON object with "
    "tempo_quarter; use null if unreadable."
)
STAFF_PROMPT = (
    'Read the first staff and its instrument label. Return JSON with instrument '
    '(guitar, bass, pitched, drums, or null if unknown) and string_count '
    '(number of TAB lines, or null if TAB is absent), and name (the visible '
    'instrument label, preserving abbreviations, or null if absent). Use the percussion clef '
    'for drums. Use pitched for other melodic instruments or unlabelled '
    'melodic notation; use null for unnamed TAB. Do not infer a '
    'string count from the five lines of standard notation.'
)
CLEF_PROMPT = (
    'Read the visible clef. Return JSON with clef (G2, F4, C3, C4, percussion, tab, '
    'or null) and clef_octave (0, -12, 12, -24, 24, or null if unreadable). '
    'A small 8 or 15 below the clef lowers its pitches; above raises them. '
    'Use tab with clef_octave 0 for a TAB symbol.'
)
ANNOTATION_PROMPT = (
    'Classify and read the visible score annotation. Return JSON with kind '
    '(instrument, ottava, capo, chord, chord_diagram, technique, tempo, title, credit, other, '
    'or null if unreadable), semitones (sounding minus written pitch, or null), '
    'capo (fret number, or null), and text (visible words, or null). '
    'A chord name such as Bb or F# and a chord fingering diagram are NOT '
    'transposition instructions. Let ring, P.M., vibrato and their continuation '
    'lines are techniques, NOT ottava. Use title for a song title and credit for a composer, arranger, artist or studio credit; preserve the original language. Use null pitch fields for non-pitch annotations. '
    'For ottava use 12 for 8va, -12 for 8vb, 24 for 15ma, -24 for 15mb. '
    'For instrument transposition use only an explicit instruction or an '
    'unambiguous instrument label. Do not treat a key signature or tuning as '
    'an instrument transposition. For chord_diagram also return diagram with '
    'base_fret, frets (absolute fret numbers, 0 open, "x" muted, LOW to HIGH string), '
    'fingers (visible numbers or null per string), barres ([absolute fret, zero-based '
    'low string index, high string index]). Preserve chord accidentals and slash bass. '
    'Read only visible dots and finger numbers; never derive a fingering from the chord name. '
    'Transcribe the evidence; do not invent missing words.'
)
TRANSPOSITION_PROMPT = ANNOTATION_PROMPT


SIGNATURE_PROMPT = (
    "Read only the printed time and key signatures in this measure image. "
    "Return S2 time=N/D key=K. Use - for a signature that is not printed. "
    "K is the signed number of key-signature accidentals: sharps positive, flats negative, "
    "cancellation to C major 0. Do not count accidentals attached to individual notes."
)


def signature_target(time_signature=None, key=None):
    return f"S2 time={time_signature or '-'} key={key if key is not None else '-'}"


def parse_signature(text):
    match = re.search(r"S2\s+time=(-|\d{1,2}/\d{1,2})\s+key=([+-]?\d|-)(?!\d)", text)
    if not match:
        raise ValueError("Invalid printed signature output")
    time, key = match.groups()
    if time != "-":
        n, d = map(int, time.split("/"))
        if not 1 <= n <= 32 or d not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("Invalid time signature")
    if key != "-" and not -7 <= int(key) <= 7:
        raise ValueError("Invalid key signature")
    return {"time": None if time == "-" else time, "key": None if key == "-" else int(key)}


def key_fifths(name):
    from scorelib.pitch_context import key_signature

    fifths = key_signature(str(name)).value[0]
    # Native engraving spells the theoretical eight-accidental keys using
    # their ordinary enharmonic equivalents (F-flat -> E, G-sharp -> A-flat).
    return fifths + 12 if fifths < -7 else fifths - 12 if fifths > 7 else fifths


def observation_prompt(mode):
    """Use the exact visible-observation prompt shared with product inference."""
    from scorelib import _native

    return _native.observation_prompt(mode)
