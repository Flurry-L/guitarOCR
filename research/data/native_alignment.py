"""Align supervision with the score actually imported and rendered by Guitar Pro."""

from fractions import Fraction
from functools import lru_cache
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
from urllib.parse import quote

from scorelib.m2 import parse_measure_target
from scorelib.m2 import format_measure_target
from scorelib.percussion import DRUM_KEY_ALIASES


def align_printed_chords(result, folder):
    """The engraver can print Cmaj7 while retaining C7M in its native model.

    Match visible names in the source measure's own rectangle to its existing
    chord onsets. Require equal counts and matching roots; never invent timing
    or use OCR predictions as supervision.
    """
    from research.data.native_chords import chord_words
    from research.common.pdf import open_pdf

    layout_path, pdf_path = folder / 'layout.json', folder / 'score.pdf'
    if not layout_path.is_file() or not pdf_path.is_file():
        return
    candidates = {index for index, measure in result.items()
                  if any(event['chord'] for event in measure['events'].values())}
    if not candidates:
        return

    def root(name):
        match = re.match(r'([A-GH])([#b]?)', str(name).replace('♭', 'b').replace('♯', '#'))
        if not match:
            return None
        return ({'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11, 'H': 11}[match[1]]
                + {'': 0, '#': 1, 'b': -1}[match[2]]) % 12

    layout = json.loads(layout_path.read_text())
    with open_pdf(pdf_path) as pdf:
        words = {}
        for system in layout['systems']:
            page = system['page']
            for box in system['measure_boxes']:
                index = box['measure_index']
                if index not in candidates:
                    continue
                if page not in words:
                    words[page] = chord_words(pdf[page - 1])
                x, y, width, height = [v * 72 / 25.4 for v in box['bbox_mm']]
                visible = sorted((word for word in words[page]
                                  if x - 1 <= (word[0] + word[2]) / 2 <= x + width + 1
                                  and y - 1 <= (word[1] + word[3]) / 2 <= y + height + 1), key=lambda w: w[0])
                events = sorted(((at, event) for at, event in result[index]['events'].items()
                                 if event['chord']), key=lambda pair: (pair[0][1], pair[0][0]))
                if len(visible) != len(events) or not all(
                    root(word[4]) == root(event['chord']) for word, (_, event) in zip(visible, events)
                ):
                    continue
                for word, (_, event) in zip(visible, events):
                    event['chord'] = word[4]


@lru_cache(maxsize=512)
def native_text(label_path, mode):
    """Read text from the model that produced the pixels, once per source/mode.

    Legacy GP files can be decoded differently by the source parser and the
    engraver. Note/rhythm targets are unaffected by this text-only alignment.
    """
    path = Path(label_path)
    root = path.parent.parent
    siblings = {'pitch_training': 'pitch_native', 'pitch_named': 'pitch_named_native',
                'parallel_synthetic': 'parallel_synthetic_native',
                'parallel_techniques': 'parallel_techniques_native'}
    export = root.parent / siblings[root.name] if root.name in siblings else root / 'native-export'
    folder = export / 'documents' / f'{mode}-{path.stem}'
    exports = list(folder.glob('tracks/*/official-score.json'))
    if len(exports) != 1:
        return {}
    model = json.loads(exports[0].read_text())
    tracks = model.get('tracks', [])
    if len(tracks) != 1 or len(tracks[0]['staves']) != 1:
        return {}
    sections = {m['master_measure_index']: m.get('section_text') or m.get('section_letter')
                for m in model['document']['master_measures']}
    result = {}
    for measure in tracks[0]['staves'][0]['measures']:
        index = measure['measure_index']
        events = {}
        for voice in measure['voices']:
            for event in voice['events']:
                if event['grace'] or event['placeholder']:
                    continue
                # Native offsets are quarter-note fractions; M2 uses 960
                # ticks per quarter, including tuplets.
                onset = round(Fraction(*event['offset']) * 960)
                events[voice['voice_index'], onset] = {
                    'text': event.get('free_text'), 'chord': event.get('chord'),
                }
        result[index] = {'section': sections.get(index), 'events': events}
    align_printed_chords(result, exports[0].parent)
    return result


def native_text_target(target, label_path, mode, measure_index):
    reference = native_text(str(label_path), mode).get(int(measure_index))
    if reference is None:
        return target
    measure = parse_measure_target(target)
    changed = measure.get('section') != reference['section']
    measure['section'] = reference['section']
    for voice in measure['voices']:
        for event in voice['events']:
            text = reference['events'].get((voice['voice'], event['start']))
            if text is None:
                continue
            previous = list(event.get('effects') or [])
            effects = [e for e in previous if not e.startswith(('text:', 'chord:'))]
            effects.extend(f'{key}:{quote(str(value).strip(), safe="")}'
                           for key, value in text.items() if value and str(value).strip())
            if sorted(previous) != sorted(effects):
                event['effects'] = effects
                changed = True
    return format_measure_target(measure, mode) if changed else target


def note_differences(label, native):
    if len(native["tracks"]) != 1 or len(native["tracks"][0]["staves"]) != 1:
        raise ValueError("Single-staff labels require one native track and staff")
    staff = native["tracks"][0]["staves"][0]
    tuning = staff["tuning_pitches_low_to_high"]
    percussion = label["track"].get("instrument") == "drums"
    capo = 0 if percussion else label["track"].get("capo", 0)
    expected, rendered = {}, {}
    for index, measure in enumerate(label["measures"]):
        for voice in parse_measure_target(measure["targets"]["notation"])["voices"]:
            for event in voice["events"]:
                pitches = sorted(
                    n["pitch"] + capo
                    for n in event.get("notes", [])
                    if "dead" not in n.get("effects", [])
                )
                if pitches:
                    expected[index, voice["voice"], event["start"]] = pitches
    for measure in staff["measures"]:
        for voice in measure["voices"]:
            for event in voice["events"]:
                if event["grace"] or event["rest"] or event["placeholder"]:
                    continue
                pitches = []
                for note in event["notes"]:
                    if note["dead"]:
                        continue
                    pitch = tuning[note["native_string_index"]] + note["fret"]
                    if not percussion:
                        pitch += staff["total_capo_frets_low_to_high"][note["native_string_index"]]
                    # Retain unsupported native values so an import that turns
                    # a valid source drum into an invisible note is rejected.
                    pitches.append(
                        DRUM_KEY_ALIASES.get(pitch, pitch) if percussion else pitch
                    )
                if pitches:
                    onset = int(Fraction(*event["offset"]) * 960)
                    rendered[measure["measure_index"], voice["voice_index"], onset] = (
                        sorted(pitches)
                    )
    return [
        {"location": key, "source": expected.get(key), "native": rendered.get(key)}
        for key in sorted(expected.keys() | rendered.keys())
        if expected.get(key) != rendered.get(key)
    ]


def verify_source(job):
    root, row = job
    label = json.loads((root / "labels" / (row["source_id"] + ".json")).read_text())
    folder = (
        root
        / "native-export"
        / "documents"
        / ("notation-" + row["source_id"])
        / "tracks"
    )
    paths = list(folder.glob("*/official-score.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one native notation export: {row['source_id']}")
    differences = note_differences(label, json.loads(paths[0].read_text()))
    return {
        "source_id": row["source_id"],
        "split": row["split"],
        "instrument": row.get("instrument", "guitar"),
        "different_events": len(differences),
        "examples": differences[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = json.loads((args.source / "source_catalog.json").read_text())["sources"]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(
            pool.map(verify_source, ((args.source, row) for row in rows), chunksize=8)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(
        {
            "sources": len(results),
            "excluded_sources": sum(bool(row["different_events"]) for row in results),
            "different_events": sum(row["different_events"] for row in results),
        }
    )


if __name__ == "__main__":
    main()
