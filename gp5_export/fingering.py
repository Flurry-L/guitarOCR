from __future__ import annotations

from collections import Counter
from typing import Any
from shared.techniques import ornament_pitches


def _choose_position(pitch: int, tuning: list[int], used_strings: set[int]) -> tuple[int, int]:
    candidates = []
    for string, open_pitch in enumerate(tuning, start=1):
        fret = pitch - open_pitch
        if 0 <= fret <= 24:
            collision = 1 if string in used_strings else 0
            candidates.append((collision, abs(fret - 5), fret, string))
    if not candidates:
        return 1, max(0, min(24, pitch - tuning[0]))
    _collision, _position_cost, fret, string = min(candidates)
    return string, fret


def _assign_positions(
    notes: list[dict[str, Any]],
    tuning: list[int],
    previous: dict[int, tuple[int, int]],
    reservations: dict[int, int],
) -> list[tuple[int, int]]:
    """Assign a collision-free guitar position to a notation-only chord."""

    candidates: list[list[tuple[float, int, int]]] = []
    for note in notes:
        if "string" in note:
            string = int(note["string"])
            fret_value = note.get("fret", 0)
            if fret_value == "x" and "pitch" in note:
                fret = max(0, int(note["pitch"]) - tuning[string - 1])
            else:
                fret = 0 if fret_value == "x" else int(fret_value)
            candidates.append([(0.0, string, fret)])
            continue
        pitch = int(note["pitch"])
        tie = "tie" in (note.get("effects") or [])
        values = []
        for string, open_pitch in enumerate(tuning, start=1):
            fret = pitch - open_pitch
            if not 0 <= fret <= 36 or any(not 0 <= p - open_pitch <= 36 for p in ornament_pitches(note)):
                continue
            previous_pitch, previous_fret = previous.get(string, (-10_000, -1))
            tie_match = tie and previous_pitch == pitch
            reservation_owner = reservations.get(string)
            reserved_match = tie and reservation_owner == pitch
            cost = abs(fret - 5) + (0 if not tie or tie_match else 1_000)
            if reservation_owner is not None and (
                reservation_owner != pitch or not tie
            ):
                cost += 2_000
            if tie and reservations and not reserved_match:
                cost += 2_000
            if tie_match:
                cost -= 1_000
                fret = previous_fret
            if reserved_match:
                cost -= 4_000
                if previous_pitch == pitch:
                    fret = previous_fret
            values.append((float(cost), string, fret))
        if not values:
            raise ValueError(f"Pitch {pitch} or its ornament cannot be played with the current tuning")
        candidates.append(sorted(values))

    order = sorted(range(len(notes)), key=lambda index: (len(candidates[index]), index))
    best: tuple[float, list[tuple[int, int]]] | None = None
    assigned: list[tuple[int, int] | None] = [None] * len(notes)

    def search(position: int, used_strings: set[int], cost: float) -> None:
        nonlocal best
        if best is not None and cost >= best[0]:
            return
        if position == len(order):
            best = (cost, [value for value in assigned if value is not None])
            return
        note_index = order[position]
        for candidate_cost, string, fret in candidates[note_index]:
            if string in used_strings:
                continue
            assigned[note_index] = (string, fret)
            search(position + 1, used_strings | {string}, cost + candidate_cost)
            assigned[note_index] = None

    search(0, set(), 0.0)
    if best is not None:
        return best[1]

    raise ValueError("Chord cannot be assigned to distinct strings; correct pitches or tuning")


def _tie_reservation_note_ids(
    measures: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """Mark the prior note that each notation-only tie must continue."""

    result: dict[int, set[int]] = {0: set(), 1: set()}
    # A chord may contain the same written pitch on multiple strings. Keep a
    # small stack of live anchors per pitch instead of collapsing them to the
    # last note; otherwise the second tied unison is assigned to an unrelated
    # string and PyGuitarPro resolves it to the wrong pitch on readback.
    previous_by_voice: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}
    for measure in measures:
        for voice in measure.get("voices", []):
            voice_index = int(voice["voice"])
            previous_by_pitch = previous_by_voice.setdefault(voice_index, {})
            for event in voice.get("events", []):
                grouped: dict[int, list[dict[str, Any]]] = {}
                for note in event.get("notes", []):
                    if "pitch" in note:
                        grouped.setdefault(int(note["pitch"]), []).append(note)
                for pitch, notes in grouped.items():
                    anchors = list(previous_by_pitch.get(pitch, []))
                    tie_count = sum(
                        "tie" in (note.get("effects") or []) for note in notes
                    )
                    if tie_count:
                        result.setdefault(voice_index, set()).update(
                            anchors[-tie_count:]
                        )
                    # Ties replace the matched anchors on their strings while
                    # older unmatched unisons may still be sounding. Cap at
                    # the physical string count to keep malformed files sane.
                    retained = anchors[:-tie_count] if tie_count else anchors
                    previous_by_pitch[pitch] = (
                        retained + [id(note) for note in notes]
                    )[-6:]
    return result


def _plan_notation_voice_positions(
    measures: list[dict[str, Any]],
    voice_index: int,
    tuning: list[int],
    *,
    beam_width: int = 256,
) -> dict[int, tuple[int, int]]:
    """Globally assign hidden strings while preserving every visible tie."""

    events: list[list[dict[str, Any]]] = []
    for measure in measures:
        voice = next(
            (
                value for value in measure.get("voices", [])
                if int(value["voice"]) == voice_index
            ),
            None,
        )
        if voice is not None:
            events.extend(list(event.get("notes", [])) for event in voice["events"])
    if not events:
        return {}
    if not any(
        "tie" in (note.get("effects") or [])
        for notes in events for note in notes
    ):
        return {}

    # At each event boundary, retain enough copies of a pitch when its next
    # occurrence is a tie.  If the next occurrence is a normal note, that
    # note can establish a new anchor and no reservation is required yet.
    required_after: list[Counter[int]] = [Counter() for _ in events]
    next_requirement: Counter[int] = Counter()
    for event_index in range(len(events) - 1, -1, -1):
        required_after[event_index] = next_requirement.copy()
        grouped: dict[int, list[dict[str, Any]]] = {}
        for note in events[event_index]:
            if "pitch" in note:
                grouped.setdefault(int(note["pitch"]), []).append(note)
        for pitch, notes in grouped.items():
            tie_count = sum(
                "tie" in (note.get("effects") or []) for note in notes
            )
            next_requirement[pitch] = tie_count if tie_count == len(notes) else 0
            if not next_requirement[pitch]:
                del next_requirement[pitch]

    # Each retained node stores only a back-pointer and this event's chosen
    # positions; paths are reconstructed once after the final event.
    nodes: list[tuple[int, list[tuple[int, tuple[int, int]]]]] = [(-1, [])]
    initial_state: tuple[int | None, ...] = tuple(None for _ in tuning)
    beams: dict[tuple[int | None, ...], tuple[float, int]] = {
        initial_state: (0.0, 0)
    }

    for event_index, notes in enumerate(events):
        candidates_by_state: dict[
            tuple[int | None, ...],
            tuple[float, int, list[tuple[int, tuple[int, int]]]],
        ] = {}
        for state, (base_cost, parent_node) in beams.items():
            note_candidates: list[list[tuple[float, int, int]]] = []
            valid = True
            for note in notes:
                pitch = int(note["pitch"])
                tie = "tie" in (note.get("effects") or [])
                values = []
                for string, open_pitch in enumerate(tuning, start=1):
                    fret = pitch - open_pitch
                    if not 0 <= fret <= 36 or any(not 0 <= p - open_pitch <= 36 for p in ornament_pitches(note)):
                        continue
                    if tie and state[string - 1] != pitch:
                        continue
                    values.append((abs(fret - 5) * 0.01, string, fret))
                if not values:
                    valid = False
                    break
                note_candidates.append(values)
            if not valid:
                continue

            order = sorted(
                range(len(notes)), key=lambda index: (len(note_candidates[index]), index)
            )
            positions: list[tuple[int, int] | None] = [None] * len(notes)

            def enumerate_assignments(
                order_index: int, used: set[int], local_cost: float
            ) -> None:
                if order_index == len(order):
                    new_state = list(state)
                    chosen = []
                    for note, position in zip(notes, positions):
                        assert position is not None
                        string, fret = position
                        pitch = int(note["pitch"])
                        new_state[string - 1] = pitch
                        chosen.append((id(note), (string, fret)))
                    # Only pitches whose next occurrence is a tie affect a
                    # future decision. Keeping every ordinary pitch in the
                    # beam creates hundreds of musically equivalent states
                    # and makes dense notation-only songs needlessly slow.
                    required_pitches = set(required_after[event_index])
                    new_state = [
                        value if value in required_pitches else None
                        for value in new_state
                    ]
                    state_tuple = tuple(new_state)
                    counts = Counter(value for value in state_tuple if value is not None)
                    if any(
                        counts[pitch] < count
                        for pitch, count in required_after[event_index].items()
                    ):
                        return
                    total_cost = base_cost + local_cost
                    previous = candidates_by_state.get(state_tuple)
                    if previous is None or total_cost < previous[0]:
                        candidates_by_state[state_tuple] = (
                            total_cost, parent_node, chosen
                        )
                    return
                note_index = order[order_index]
                for cost, string, fret in note_candidates[note_index]:
                    if string in used:
                        continue
                    positions[note_index] = (string, fret)
                    enumerate_assignments(
                        order_index + 1, used | {string}, local_cost + cost
                    )
                    positions[note_index] = None

            enumerate_assignments(0, set(), 0.0)

        if not candidates_by_state:
            return {}
        ranked = sorted(
            candidates_by_state.items(), key=lambda item: (item[1][0], repr(item[0]))
        )[:beam_width]
        beams = {}
        for state, (cost, parent_node, chosen) in ranked:
            nodes.append((parent_node, chosen))
            beams[state] = (cost, len(nodes) - 1)

    _cost, node_index = min(beams.values(), key=lambda value: value[0])
    plan: dict[int, tuple[int, int]] = {}
    while node_index > 0:
        parent_node, chosen = nodes[node_index]
        plan.update(chosen)
        node_index = parent_node
    return plan
