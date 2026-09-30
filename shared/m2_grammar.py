"""Mode-specific M2 syntax for verified, constrained generation."""

from functools import lru_cache
import json


def alternatives(values):
    return ' | '.join(json.dumps(str(v)) for v in values)


def onset_rule(ticks):
    """Compact decimal grammar for onsets strictly inside a known measure."""
    if ticks is None:
        return 'integer'
    limit = str(max(0, int(ticks) - 1))
    choices = ['"0"']
    for size in range(1, len(limit)):
        choices.append('[1-9]' + (f' [0-9]{{{size - 1}}}' if size > 1 else ''))
    for index, digit in enumerate(limit):
        low, high = (1 if index == 0 else 0), int(digit) - 1
        if low <= high:
            prefix = json.dumps(limit[:index]) + ' ' if index else ''
            tail = len(limit) - index - 1
            choices.append(prefix + f'[{low}-{high}]' + (f' [0-9]{{{tail}}}' if tail else ''))
    choices.append(json.dumps(limit))
    return ' | '.join(choices)


@lru_cache(maxsize=256)
def measure_grammar(mode, strings=12, measure_ticks=None):
    from guitarpro.models import KeySignature

    if mode not in {'notation', 'tab', 'both'}:
        raise ValueError('Unknown M2 display mode')
    fields = {'notation': 'pitch', 'tab': 'position', 'both': 'position pitch'}[mode]
    return r'''
root ::= "M2" time? tempo? key? feel? bars? repeat? alternate? section? direction? from-direction? " | " voice (" || " voice)*
time ::= " time=" positive "/" denominator
tempo ::= " tempo=" positive
key ::= " key=" key-name
feel ::= " feel=" word
bars ::= " bar=" word ("," word)*
repeat ::= " rep=" positive
alternate ::= " alt=" positive
section ::= " section=" word
direction ::= " dir=" word
from-direction ::= " from=" word
voice ::= "V" voice-id "{" event (" " (event | repeat-event)){0,255} "}"
event ::= "@" onset ":" duration ":" payload
repeat-event ::= "@" onset ":" duration ":^"
duration ::= ("w" | "h" | "q" | "e" | "s" | "t" | "f" | "d128") ("." | "..")? ("[" tuplet ":" tuplet "]")?
payload ::= ("r" | "e" | "z" | note ("," note)*) ("<" beat-effect ("," beat-effect)* ">")?
note ::= NOTE_FIELDS ("(" note-effect ("," note-effect)* ")")?
position ::= "s" string-id "f" (fret | "x")
pitch ::= "p" midi
note-effect ::= "accent" | "dead" | "ghost" | "hammer" | "heavy" | "let" | "pm" | "sia" | "sib" | "sl" | "sod" | "sou" | "ss" | "stacc" | "tap" | "tie" | "vib" | "accswap" | "vel:" midi | "harm:" word | "slide:" word | "bend:" word | "grace" (":" word)? | "trill" (":" word)? | "trem" (":" word)?
beat-effect ::= "fade" | "pick_down" | "pick_up" | "rasg" | "stroke_down" | "stroke_up" | "ottava:" ("12" | "-12" | "24" | "-24") | "tempo:" positive | "dyn:" midi | "chord:" chord-name | "diagram:" word | "slap:" ("none" | "tapping" | "slapping" | "popping") | "text:" word
# Chord fields must not become an escape route for arbitrary prose when a
# damaged measure is retried. The unconstrained first pass and IR still retain
# custom printed names; this rule only guides constrained regeneration.
chord-name ::= chord-root chord-quality? chord-extension? chord-post-quality? chord-modifier{0,4} chord-group? chord-bass? | "N.C." | "N.C" | "NC" | "n.c." | "No%20Chord"
chord-root ::= [A-Ha-h] accidental? | accidental? ("I" | "II" | "III" | "IV" | "V" | "VI" | "VII" | "i" | "ii" | "iii" | "iv" | "v" | "vi" | "vii" | [1-7])
accidental ::= "#" | "b" | "##" | "bb" | "x" | "%23" | "%23%23" | "%E2%99%AD" | "%E2%99%AF" | "♭" | "♯"
chord-quality ::= "m" | "M" | "maj" | "Maj" | "min" | "Min" | "dim" | "aug" | "sus" | "dom" | "+" | "%2B" | "-" | "o" | "°" | "%C2%B0" | "ø" | "%C3%B8" | "Δ" | "%CE%94"
chord-extension ::= "2" | "4" | "5" | "6" | "7" | "9" | "11" | "13"
chord-post-quality ::= "M" | "+" | "%2B" | "-" | "°" | "%C2%B0" | "ø" | "%C3%B8"
chord-modifier ::= ("#" | "b" | "%23" | "%E2%99%AD" | "%E2%99%AF" | "♭" | "♯") ("5" | "6" | "7" | "9" | "11" | "13") | ("maj" | "Maj" | "M") ("7" | "9" | "11" | "13") | ("sus" | "add" | "omit" | "no") ("2" | "3" | "4" | "5" | "6" | "7" | "9" | "11" | "13") | "alt"
chord-group ::= "%28" (chord-modifier | chord-extension) ("%2C"? (chord-modifier | chord-extension)){0,3} "%29"
chord-bass ::= ("/" | "%2F") (chord-root | "9" | "11" | "13")
# Constrained retries bound individual fields, not the whole measure. Long
# free text recognized successfully in the first pass remains untouched.
word ::= nonspace (nonspace | "%20" nonspace){0,127}
nonspace ::= [^ \t\r\n,()<>|{}%] | "%" ([013-9A-Fa-f] hex | "2" [1-9A-Fa-f])
hex ::= [0-9A-Fa-f]
integer ::= "0" | [1-9] [0-9]{0,6}
positive ::= [1-9] [0-9]{0,6}
denominator ::= "1" | "2" | "4" | "8" | "16" | "32" | "64"
'''.replace('NOTE_FIELDS', fields) + '\n'.join([
        'onset ::= ' + onset_rule(measure_ticks),
        'voice-id ::= ' + alternatives(range(16)),
        'string-id ::= ' + alternatives(range(1, min(12, max(1, strings)) + 1)),
        'fret ::= ' + alternatives(range(37)),
        'midi ::= ' + alternatives(range(128)),
        'tuplet ::= ' + alternatives(range(1, 33)),
        'key-name ::= ' + alternatives(k.name for k in KeySignature),
    ]) + '\n'
