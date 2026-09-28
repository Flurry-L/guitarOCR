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
        if any(len(e.get('notes', [])) > 7 for m in measures for v in m['voices'] for e in v['events']):
            raise ValueError('Percussion chord requires more than seven GP5 slots')
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
        # Coordinate descent can get stuck when several slots must move
        # together. Intervals have a consecutive-neighbourhood matching:
        # every interval must contain enough slots for all notes whose
        # admissible bases lie entirely inside it.
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix

        requirements = {}
        for chord in chords:
            intervals = [(max(0, high - MAX_MELODIC_FRET), low) for low, high in chord]
            for left, _ in intervals:
                for _, right in intervals:
                    if left <= right:
                        demand = sum(left <= a <= b <= right for a, b in intervals)
                        requirements[left, right] = max(requirements.get((left, right), 0), demand)
        matrix = [[int(left <= base <= right) for base in range(128)] for left, right in requirements]
        matrix.append([1] * 128)
        solution = milp(np.ones(128), integrality=np.ones(128), bounds=Bounds(0, 7),
                        constraints=LinearConstraint(csr_matrix(matrix, dtype=float),
                                                     [*requirements.values(), 0],
                                                     [*[np.inf] * len(requirements), 7]),
                        options={'time_limit': 5})
        if solution.x is not None:
            bases = [base for base, count in enumerate(solution.x) for _ in range(round(count))]
            bases += [low] * (7 - len(bases))
            if len(bases) == 7 and cost(bases) == 0:
                return sorted(bases, reverse=True)
        raise ValueError('GP5 note slots cannot preserve these notes; use the score IR or MusicXML')
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
            # Excerpts may omit the tie's origin. Allocate its explicit pitch
            # normally; the writer records the detached tie in its projection.
            continue
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
