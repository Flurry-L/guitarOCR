"""Mode-specific M2 syntax for verified, constrained generation."""

from functools import lru_cache
import json


def alternatives(values):
    return ' | '.join(json.dumps(str(v)) for v in values)


@lru_cache(maxsize=36)
def measure_grammar(mode, strings=12):
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
voice ::= "V" voice-id "{" event (" " (event | repeat-event))* "}"
event ::= "@" integer ":" duration ":" payload
repeat-event ::= "@" integer ":" duration ":^"
duration ::= ("w" | "h" | "q" | "e" | "s" | "t" | "f" | "d128") ("." | "..")? ("[" tuplet ":" tuplet "]")?
payload ::= ("r" | "e" | "z" | note ("," note)*) ("<" beat-effect ("," beat-effect)* ">")?
note ::= NOTE_FIELDS ("(" note-effect ("," note-effect)* ")")?
position ::= "s" string-id "f" (fret | "x")
pitch ::= "p" midi
note-effect ::= "accent" | "dead" | "ghost" | "hammer" | "heavy" | "let" | "pm" | "sia" | "sib" | "sl" | "sod" | "sou" | "ss" | "stacc" | "tap" | "tie" | "vib" | "accswap" | "vel:" midi | "harm:" word | "slide:" word | "bend:" word | "grace" (":" word)? | "trill" (":" word)? | "trem" (":" word)?
beat-effect ::= "fade" | "pick_down" | "pick_up" | "rasg" | "stroke_down" | "stroke_up" | "ottava:" ("12" | "-12" | "24" | "-24") | "tempo:" positive | "dyn:" midi | "chord:" word | "slap:" word | "text:" word
word ::= [^ \t\r\n,()<>|{}]+
integer ::= "0" | [1-9] [0-9]*
positive ::= [1-9] [0-9]*
denominator ::= "1" | "2" | "4" | "8" | "16" | "32" | "64"
'''.replace('NOTE_FIELDS', fields) + '\n'.join([
        'voice-id ::= ' + alternatives(range(16)),
        'string-id ::= ' + alternatives(range(1, min(12, max(1, strings)) + 1)),
        'fret ::= ' + alternatives(range(37)),
        'midi ::= ' + alternatives(range(128)),
        'tuplet ::= ' + alternatives(range(1, 33)),
        'key-name ::= ' + alternatives(k.name for k in KeySignature),
    ]) + '\n'
