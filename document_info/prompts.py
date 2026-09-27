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
    '(number of TAB lines, or null if TAB is absent). Use the percussion clef '
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
TRANSPOSITION_PROMPT = (
    'Read the visible pitch instruction. Return JSON with kind (instrument, '
    'ottava, capo, or null), semitones (sounding minus written pitch, or null), '
    'capo (fret number, or null), and text (visible words, or null). '
    'For ottava use 12 for 8va, -12 for 8vb, 24 for 15ma, -24 for 15mb. '
    'For instrument transposition use only an explicit instruction or an '
    'unambiguous instrument label. Do not treat a key signature or tuning as '
    'an instrument transposition.'
)
