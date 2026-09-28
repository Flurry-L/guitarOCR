"""Store pitches in GP5's note slots without inventing guitar fingerings."""

from collections import Counter
from shared.techniques import ornament_pitches
from gp5_export.fingering import MAX_MELODIC_FRET


def _unassigned(chord, bases):
    available = sorted(bases)
    missing = 0
    for low, high in sorted(chord):
        candidates = [base for base in available if base <= low and high <= base + MAX_MELODIC_FRET]
        if candidates:
            available.remove(candidates[0])
        else:
            missing += 1
    return missing


def storage_tuning(measures, percussion=False):
    if percussion:
        return [0] * 7

    def span(note):
        values = [int(note["pitch"]), *ornament_pitches(note)]
        return min(values), max(values)

    chords = Counter(
        tuple(sorted(span(n) for n in event.get("notes", [])))
        for measure in measures
        for voice in measure["voices"]
        for event in voice["events"]
    )
    pitches = [pitch for chord in chords for interval in chord for pitch in interval]
    low, high = min(pitches, default=60), max(pitches, default=60)
    # Guitar Pro's GP5 importer silently replaces melodic frets above 30 with
    # zero, although PyGuitarPro itself accepts up to 99. Keep every stored
    # melodic offset inside the native importer's range.
    if high - low <= MAX_MELODIC_FRET:
        return [low] * 7
    candidates = sorted(
        {max(low, pitch - offset) for pitch in pitches for offset in (0, MAX_MELODIC_FRET // 2, MAX_MELODIC_FRET)}
    )
    # A base below the lowest used pitch only reduces the slot's range.
    # Keep all seven slots usable for dense chords and sustained unisons.
    bases = [max(low, high - MAX_MELODIC_FRET - 12 * index) for index in range(7)]

    def cost(values):
        return sum(
            _unassigned(chord, values) * weight for chord, weight in chords.items()
        )

    current = cost(bases)
    for _ in range(14):
        if not current:
            return bases
        best = current
        replacement = None
        for slot in range(7):
            for base in candidates:
                proposal = [*bases]
                proposal[slot] = base
                value = cost(proposal)
                if value < best:
                    best, replacement = value, proposal
        if replacement is None:
            break
        bases, current = replacement, best
    if current:
        raise ValueError(
            "These chords and pitch range cannot fit the GP5 note slots; the complete notes remain in score.json"
        )
    return bases


def pitch_positions(notes, tuning, previous, reserved):
    if len(notes) > 7:
        raise ValueError(
            "GP5 stores at most seven simultaneous notes per voice; the complete notes remain in score.json"
        )
    positions = [None] * len(notes)
    used = set()
    # Tied notes keep their slot, since GP5 encodes a tie by slot, not pitch.
    for index, note in enumerate(notes):
        if "tie" not in note.get("effects", []):
            continue
        pitch = int(note["pitch"])
        choices = [s for s, (p, _f) in previous.items() if p == pitch and s not in used]
        if not choices:
            raise ValueError(f"Tie has no preceding pitch {pitch}")
        string = min(choices)
        used.add(string)
        positions[index] = (string, pitch - tuning[string - 1])
    for index, note in enumerate(notes):
        if positions[index] is not None:
            continue
        pitch = int(note["pitch"])
        choices = [s for s in range(1, 8) if s not in used and s not in reserved]
        if not choices:
            raise ValueError(
                "GP5 note slots are occupied by sustained notes; the complete notes remain in score.json"
            )
        string = min(choices, key=lambda s: (previous.get(s, (None,))[0] != pitch, s))
        used.add(string)
        positions[index] = (string, pitch - tuning[string - 1])
    return positions
