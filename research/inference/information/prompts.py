HEADER_PROMPT = (
    "Read only visible song title, artist, and tuning label. Return one JSON object "
    "with title, artist, tuning_name; use null for absent fields."
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
STAFF_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['instrument', 'string_count', 'name'],
    'properties': {
        'instrument': {'enum': ['guitar', 'bass', 'pitched', 'drums', None]},
        'string_count': {'anyOf': [{'type': 'integer', 'minimum': 1, 'maximum': 12}, {'type': 'null'}]},
        'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
    },
}
CLEF_PROMPT = (
    'Read the visible clef. Return JSON with clef (G2, F4, C3, C4, percussion, tab, '
    'or null) and clef_octave (0, -12, 12, -24, 24, or null if unreadable). '
    'A small 8 or 15 below the clef lowers its pitches; above raises them. '
    'Use tab with clef_octave 0 for a TAB symbol.'
)
ANNOTATION_PROMPT = (
    'Classify and read the visible score annotation. Return JSON with kind '
    '(instrument, ottava, capo, chord, chord_diagram, technique, tempo, other, '
    'or null if unreadable), semitones (sounding minus written pitch, or null), '
    'capo (fret number, or null), and text (visible words, or null). '
    'A chord name such as Bb or F# and a chord fingering diagram are NOT '
    'transposition instructions. Let ring, P.M., vibrato and their continuation '
    'lines are techniques, NOT ottava. Use null pitch fields for non-pitch annotations. '
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

ANNOTATION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['kind', 'semitones', 'capo', 'text'],
    'properties': {
        'kind': {'enum': ['instrument', 'ottava', 'capo', 'chord', 'chord_diagram', 'technique', 'tempo', 'other', None]},
        'semitones': {'anyOf': [{'type': 'integer', 'minimum': -36, 'maximum': 36}, {'type': 'null'}]},
        'capo': {'anyOf': [{'type': 'integer', 'minimum': 0, 'maximum': 24}, {'type': 'null'}]},
        'text': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        'diagram': {'anyOf': [{'type': 'null'}, {
            'type': 'object', 'additionalProperties': False,
            'required': ['base_fret', 'frets', 'fingers', 'barres'],
            'properties': {
                'base_fret': {'type': 'integer', 'minimum': 1, 'maximum': 36},
                'frets': {'type': 'array', 'minItems': 3, 'maxItems': 12,
                          'items': {'anyOf': [{'type': 'integer', 'minimum': 0, 'maximum': 36}, {'const': 'x'}]}},
                'fingers': {'type': 'array', 'minItems': 3, 'maxItems': 12,
                            'items': {'enum': [None, 0, 1, 2, 3, 4]}},
                'barres': {'type': 'array', 'items': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                                                    'items': {'type': 'integer', 'minimum': 0, 'maximum': 36}}},
            },
        }]},
    },
}
