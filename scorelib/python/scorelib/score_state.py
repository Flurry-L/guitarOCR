"""Printed state and visual neighbourhoods for independent measure decoding."""

from __future__ import annotations

import json
import re

from scorelib.m2 import parse_measure_target
from scorelib.m2 import format_measure_target


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
    from guitarpro.models import KeySignature

    fifths = KeySignature[str(name)].value[0]
    # Native engraving spells the theoretical eight-accidental keys using
    # their ordinary enharmonic equivalents (F-flat -> E, G-sharp -> A-flat).
    return fifths + 12 if fifths < -7 else fifths - 12 if fifths > 7 else fifths


def state_prompt(mode, instrument, state, pitch_context=None, *, first=False, visual_pitch=False):
    from scorelib.pitch_context import prompt_pitch_context

    if mode == 'tab':
        # TAB has no printed key signature; its frets must not depend on an
        # unobservable key copied from the original score file during training.
        state = {**state, 'key': 0}
    if visual_pitch and mode != 'tab':
        state = {k: v for k, v in state.items() if k != 'tuning'}
    fields = "string, fret" if mode == "tab" else "written MIDI pitch" if mode == "notation" else (
        "string, fret, written MIDI pitch" if visual_pitch else "string, fret, MIDI pitch")
    prompt = (
        f"Independent {instrument} {mode} measure recognition. The first image is the target measure. "
        "The second and third images show the previous and next measures for visual context only. "
        f"Return exactly one M2 fragment for the FIRST image. Preserve every voice, event start, "
        f"duration, {fields}, rest, tie and visible technique. "
        "Use neighbouring images to resolve cross-bar ties and voice continuity; do not transcribe them. "
    )
    if first:
        prompt += "This is the first measure of the score. "
    prompt += "Effective printed state: " + json.dumps(state, separators=(",", ":")) + ". "
    if instrument == "drums":
        prompt += "Pitch fields are General MIDI drum keys. "
    elif mode == "notation" or (visual_pitch and mode == "both"):
        prompt += "Return written pitches, before instrument transposition, clef octave and ottava. "
        if mode == 'both':
            prompt += "Read pitches from the notation and string/fret from TAB independently. Do not assume a tuning. "
    if pitch_context and mode != "tab" and instrument != "drums":
        prompt += "Pitch context: " + json.dumps(prompt_pitch_context(pitch_context), separators=(",", ":")) + ". "
        prompt += "Preserve ottava:12, ottava:-12, ottava:24 or ottava:-24 on each affected event. "
    prompt += "Print time/key metadata only at score start or a printed change; use the effective state to read notes."
    return prompt


def measure_messages(record, state=None):
    state = state or record.get("score_state") or {"time": "4/4", "key": 0}
    if record.get("tuning"):
        state = {**state, "tuning": record["tuning"]}
    images = [record["image"], record.get("previous_image") or record["image"], record.get("next_image") or record["image"]]
    prompt = state_prompt(record["mode"], record.get("instrument", "guitar"), state,
                          record.get("pitch_context"), first=int(record.get("bar_index", record.get("measure_index", record.get("measure_number", 1) - 1))) == 0,
                          visual_pitch=record.get('visual_pitch', False))
    names = [p['parsed']['text'] for p in record.get('chord_annotations', []) if p.get('parsed', {}).get('text')]
    if names:
        import json
        prompt += f" Visible chord candidates in the FIRST measure: {json.dumps(names, ensure_ascii=False)}. Read their actual onsets from the image, including repeated names; do not invent notes or evenly spaced chord timing."
    return [{"role": "user", "content": [*({"type": "image", "url": path} for path in images),
                                            {"type": "text", "text": prompt}]}]


def attach_neighbours(records):
    groups = {}
    for row in records:
        key = (row.get("source_id", ""), row.get("part_id", "part-1"), row.get("staff_id", "staff-1"))
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: int(r.get("bar_index", r.get("measure_index", r.get("measure_number", 1) - 1))))
        for i, row in enumerate(rows):
            number = int(row.get("bar_index", row.get("measure_index", row.get("measure_number", 1) - 1)))
            for field, j in (("previous_image", i - 1), ("next_image", i + 1)):
                neighbour = rows[j] if 0 <= j < len(rows) else row
                other = int(neighbour.get("bar_index", neighbour.get("measure_index", neighbour.get("measure_number", 1) - 1)))
                row[field] = neighbour["image"] if abs(other - number) <= 1 else row["image"]
    return records


def resolve_fretted_pitches(target, tuning, *, verified_strings=None):
    """Derive redundant M2 pitches from the visible fingering and tuning."""
    from scorelib.techniques import ornament_position

    if not tuning:
        return target, False
    measure = parse_measure_target(target)
    changed = False
    for voice in measure['voices']:
        for event in voice['events']:
            for note in event.get('notes', []):
                string, fret = note.get('string'), note.get('fret')
                if not isinstance(string, int) or not 1 <= string <= len(tuning):
                    continue
                if verified_strings is not None and string not in verified_strings:
                    continue
                if fret == 'x' or any(e in {'dead', 'tie'} or e.startswith('harm:') for e in note.get('effects', [])):
                    continue
                if isinstance(fret, int) and 0 <= fret <= 36:
                    pitch = int(tuning[string - 1]) + fret
                    if 0 <= pitch <= 127 and note.get('pitch') != pitch:
                        note['pitch'] = pitch
                        changed = True
                for i, effect in enumerate(note.get('effects', [])):
                    parts = effect.split(':')
                    if parts[0] not in {'grace', 'trill'} or len(parts) < 2:
                        continue
                    ornament_fret, _ = ornament_position(parts[1])
                    if ornament_fret is None or not 0 <= ornament_fret <= 36:
                        continue
                    pitch = int(tuning[string - 1]) + ornament_fret
                    if not 0 <= pitch <= 127:
                        continue
                    position = f'f{ornament_fret}p{pitch}'
                    if parts[1] != position:
                        parts[1] = position
                        note['effects'][i] = ':'.join(parts)
                        changed = True
    return (format_measure_target(measure, 'both', preserve_playback=True), True) if changed else (target, False)


def resolve_measure_state(target, record):
    """Keep emitted metadata consistent with the separately read signatures."""
    from guitarpro.models import KeySignature

    state = record.get('score_state')
    if not state or record.get('state_needs_review'):
        return target, False
    measure = parse_measure_target(target)
    changed = False
    if measure.get('time_signature') and measure['time_signature'] != state['time']:
        measure['time_signature'] = state['time']
        changed = True
    key = measure.get('key_signature')
    if key and record['mode'] != 'tab':
        try:
            original = KeySignature[key]
            minor = original.value[1]
            fifths = key_fifths(key)
        except KeyError:
            minor, fifths = 0, None
        if fifths != state['key']:
            measure['key_signature'] = next(k.name for k in KeySignature
                if k.value == (state['key'], minor))
            changed = True
    return (format_measure_target(measure, record['mode'], preserve_playback=True), True) if changed else (target, False)


def resolve_ties(records):
    """Resolve unambiguous tie continuations in each voice after batch decoding."""
    last_by_part, indices = {}, {}

    def with_pitch(note, row):
        value = dict(note)
        tuning = row.get('tuning') or []
        if ('pitch' not in value and type(value.get('fret')) is int
                and 1 <= value.get('string', 0) <= len(tuning)):
            value['pitch'] = tuning[value['string'] - 1] + value['fret']
        return value

    for row in records:
        part = (row.get('source_id', ''), row.get("part_id", "part-1"), row.get("staff_id", "staff-1"))
        index = row.get('bar_index', row.get('measure_index'))
        if index is not None:
            if part in indices and index != indices[part] + 1:
                last_by_part.pop(part, None)
            indices[part] = index
        last = last_by_part.setdefault(part, {})
        measure = parse_measure_target(row["target"])
        changed = False
        present = {v['voice'] for v in measure['voices']}
        for voice_id in set(last) - present:
            del last[voice_id]
        for voice in measure["voices"]:
            voice_last = last.setdefault(voice["voice"], [])
            for event in voice["events"]:
                if event.get("status") in {"rest", "empty"}:
                    voice_last.clear()
                    continue
                notes = event.get("notes", [])
                for note in notes:
                    if 'tie' not in note.get('effects', []) or row.get('manually_edited'):
                        continue
                    if 'string' in note:
                        candidates = [n for n in voice_last if n.get('string') == note['string']]
                        if not candidates:
                            pitch = with_pitch(note, row).get('pitch')
                            candidates = [n for n in voice_last if 'string' not in n and pitch is not None and n.get('pitch') == pitch]
                    else:
                        candidates = [n for n in voice_last if n.get('pitch') == note.get('pitch')]
                        if not candidates and 'pitch' in note:
                            # A tie carries the previous accidental across the barline.
                            # Only repair a unique adjacent pitch; ambiguous chords stay as read.
                            candidates = [n for n in voice_last if 'pitch' in n and abs(n['pitch'] - note['pitch']) <= 2]
                    if len(candidates) == 1:
                        previous = candidates[0]
                        for field in ("fret", "pitch"):
                            if field in note and field in previous and note[field] != previous[field]:
                                note[field] = previous[field]
                                changed = True
                voice_last = [with_pitch(n, row) for n in notes]
                last[voice["voice"]] = voice_last
        if changed:
            row["target"] = format_measure_target(measure, row["mode"], preserve_playback=True)
            row["boundary_resolved"] = True
    return records
