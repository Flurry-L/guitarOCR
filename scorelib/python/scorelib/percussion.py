"""Canonical drum keys for symbols that Guitar Pro renders identically.

Native PDF raster comparisons establish these equivalences for the default
GP8 drum kit. They describe visible notation, not identical synthesized sounds.
"""

VISIBLE_DRUM_KEYS = frozenset(range(29, 88)) - {32}
DRUM_KEY_ALIASES = {33: 37, 34: 38, 39: 38, 40: 38, 41: 45}


def visible_drum_key(key: int) -> int:
    if key not in VISIBLE_DRUM_KEYS:
        raise ValueError(f"Drum key {key} has no verified visible GP8 notehead")
    return DRUM_KEY_ALIASES.get(key, key)
