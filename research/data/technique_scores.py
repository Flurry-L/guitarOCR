"""Playable fresh cadences for explicit text/span/event relation supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from scorelib.score import (
    duration_ticks,
    format_measure_target,
    parse_measure_target,
)

from research.common.techniques import event_features
from research.data.prompts import key_fifths
from research.data.score_rendering import audit, browser_page
from research.data.technique_span_relations import (
    compile_row,
    continuation_variant,
    previous_image,
)


def extract(state, geometry, page_path, root):
    systems = {}
    for mi, box in enumerate(geometry["boxes"]):
        systems.setdefault(round(box[1], 3), []).append(mi)
    visible = {}
    for span in geometry["techniqueSpans"]:
        if span["coordinate_space"] != "host_screenshot_pixels":
            raise ValueError("Effect bounds must match screenshot pixels")
        for event in span["events"]:
            visible.setdefault((event["mi"], event["vi"], event["ei"]), set()).add(
                span["kind"]
            )
    rows = []
    with Image.open(page_path) as page:
        for si, measures in enumerate(systems.values()):
            boxes = [geometry["boxes"][mi] for mi in measures]
            crop = [
                max(0, int(min(b[0] for b in boxes)) - 4),
                max(0, int(min(b[1] for b in boxes)) - 4),
                min(page.width, int(max(b[0] + b[2] for b in boxes)) + 5),
                min(page.height, int(max(b[1] + b[3] for b in boxes)) + 5),
            ]
            image = page.crop(crop)
            path = root / f"system-{si}.png"
            image.save(path)
            events = []
            for anchor in geometry["eventAnchors"]:
                mi, ei = anchor["measure"], anchor["event_index"]
                if mi not in measures:
                    continue
                vi = next(
                    i
                    for i, v in enumerate(state["measures"][mi]["parsed"]["voices"])
                    if v["voice"] == anchor["voice"]
                )
                kinds = visible.get((mi, vi, ei), set())
                event = state["measures"][mi]["parsed"]["voices"][vi]["events"][ei]
                authored = {
                    effect
                    for note in event["notes"]
                    for effect in note["effects"]
                    if effect in ("pm", "let")
                }
                if kinds != authored:
                    raise ValueError(
                        f"Unpainted or wrongly linked effect: {state['id']}/{mi}/{ei}"
                    )
                views = anchor["views"]
                bounds = views[-1 if state["mode"] == "both" else 0]["bbox"]
                events.append(
                    {
                        "measure_index": mi,
                        "voice": anchor["voice"],
                        "onset": anchor["onset"],
                        "event_index": ei,
                        "pixel": [
                            bounds[0] + bounds[2] / 2 - crop[0],
                            bounds[1] + bounds[3] / 2 - crop[1],
                        ],
                        "label": "both"
                        if len(kinds) == 2
                        else next(iter(kinds), "none"),
                    }
                )
            rows.append(
                {
                    "id": f"{state['id']}-{si}",
                    "image": str(path),
                    "source_page": str(page_path),
                    "bbox": crop,
                    "mode": state["mode"],
                    "events": events,
                    "measures": [
                        {
                            "measure_index": mi,
                            "target": state["measures"][mi]["target"],
                            "bbox": geometry["boxes"][mi],
                        }
                        for mi in measures
                    ],
                    "label_source": "painted effect glyph linked beats; all authored notes share each effect",
                }
            )
    return rows


COMPOSITIONS = (
    ("Orchard Walk", "CMajor", "C F Am G7"),
    ("Willow Gate", "GMajor", "Em C Am D7"),
    ("Coast Road", "DMajor", "Bm G Em A7"),
    ("Second Sunrise", "AMajor", "F#m D Bm E7"),
    ("Evening Footbridge", "EMajor", "C#m A F#m B7"),
    ("Candle Room", "AMinor", "Am C F E7"),
    ("Rain on Glass", "EMinor", "Em G C B7"),
    ("September Waltz", "DMinor", "Dm F Bb A7"),
    ("Bookshop Window", "FMajor", "Dm Bb Gm C7"),
    ("Harbour Bells", "BbMajor", "Gm Eb Cm F7"),
    ("Back Garden", "CMajor", "Fmaj7 C Dm G7"),
    ("Snow Lantern", "GMajor", "Cadd9 G Am D7"),
    ("Winding Brook", "DMajor", "G D Em A7"),
    ("Small Courtyard", "AMajor", "D A Bm E7"),
    ("Dusk Sketch", "EMinor", "C Am B7 Em"),
    ("Open Notebook", "AMinor", "F Dm E7 Am"),
    ("Maple Avenue", "CMajor", "Am F Dm G7"),
    ("Across the Lake", "GMajor", "Am C D7 G"),
    ("Night Ferry", "DMinor", "Bb Gm A7 Dm"),
    ("Quiet Return", "FMajor", "Gm Bb C7 F"),
    ("Riverside Steps", "AMajor", "F#m Bm E7 A"),
    ("Blue Hour", "EMajor", "A F#m B7 E"),
    ("Paper Boats", "DMinor", "Gm Dm Bb A7"),
    ("Warm Train", "BbMajor", "Cm Eb F7 Bb"),
    ("After the Rain", "CMajor", "Fmaj7 Dm G7 C"),
    ("Misty Hills", "GMajor", "Em Am D7 G"),
    ("April Courtyard", "DMajor", "D G Bm A7"),
    ("Golden Leaves", "GMajor", "G C Am D7"),
    ("Starlit Steps", "AMinor", "Am Dm G E7"),
    ("Porch Waltz", "FMajor", "F Dm Bb C7"),
)


SHAPES = {'C': [0, 1, 0, 2, 3, None],
 'Am': [0, 1, 2, 2, 0, None],
 'F': [1, 1, 2, 3, None, None],
 'G': [3, 0, 0, 0, 2, 3],
 'Dm': [1, 3, 2, 0, None, None],
 'Gm': [3, 3, 3, 5, 5, 3],
 'Bb': [1, 3, 3, 3, 1, None],
 'A7': [0, 2, 0, 2, 0, None],
 'Em': [0, 0, 0, 2, 2, 0],
 'B7': [2, 0, 2, 1, 2, None],
 'F#m': [2, 2, 2, 4, 4, 2],
 'Bm': [2, 3, 4, 4, 2, None],
 'C#m': [4, 5, 6, 6, 4, None],
 'E': [0, 0, 1, 2, 2, 0],
 'A': [0, 2, 2, 2, 0, None],
 'D': [2, 3, 2, 0, None, None],
 'G7': [1, 0, 0, 0, 2, 3],
 'D7': [2, 1, 2, 0, None, None],
 'E7': [0, 0, 1, 0, 2, 0],
 'C7': [0, 1, 3, 2, 3, None],
 'F7': [1, 1, 2, 1, 3, 1],
 'Cm': [3, 4, 5, 5, 3, None],
 'Fmaj7': [0, 1, 2, 3, None, None],
 'Gmaj7': [2, 0, 0, 0, 2, 3],
 'Cadd9': [3, 3, 0, 2, 3, None],
 'Asus2': [0, 0, 2, 2, 0, None],
 'Dsus4': [3, 3, 2, 0, None, None],
 'Eb': [3, 4, 3, 5, 6, None]}

RHYTHMS = (('4/4', (8, 8, 8, 8, 8, 8, 8, 8)),
 ('4/4', (4, 8, 8, 4, 8, 8)),
 ('4/4', (8, 4, 8, 8, 8, 4)),
 ('4/4', (4, 4, 4, 4)),
 ('3/4', (4, 8, 8, 4)),
 ('6/8', (8, 8, 8, 8, 8, 8)))


PICKING = (
    (0, 1, -1, 2, -2, 1, 2, -1),
    (0, -2, 1, -1, 2, 1, -2, -1),
    (0, 2, 1, -2, -1, 1, 2, -2),
    (0, -1, -2, 1, 2, -1, 1, -2),
)


def recipe(family, mode, variant, layout):
    title, key, progression = COMPOSITIONS[family]
    split = "train" if family < 12 else "validation" if family < 16 else "test"
    identity = f"technique-{family:02d}"
    case = f"{identity}-{mode}-{variant}-{layout}"
    meter, rhythm = RHYTHMS[(family * 5 + 2) % len(RHYTHMS)]
    tuning = [64, 59, 55, 50, 45, 38 if family % 4 == 1 else 40]
    rows = []
    for bar, chord in enumerate(progression.split()):
        shape = SHAPES[chord]
        strings = [s for s in range(6, 0, -1) if shape[s - 1] is not None]
        events, tick = [], 0
        for ei, duration in enumerate(rhythm):
            picked = [strings[PICKING[family % 4][(ei + bar) % 8]]]
            if family % 3 == 1 and ei % 3 == 0:
                picked = strings[-3:]
            if variant == 3 and ei % 2 == 0:
                picked = [strings[0], *strings[-2:]]
            count = len(rhythm)
            if variant == 0:
                effect = ("pm" if bar < 2 else "let") if ei < count - 1 else None
            elif variant == 1:
                effect = "let" if ei < 2 else "pm" if ei < count - 1 else None
            elif variant == 2:
                effect = "pm" if ei < 2 else "let" if ei < count - 1 else None
            elif variant == 3:
                effect = (
                    ("pm" if bar % 2 == 0 else "let") if 0 < ei < count - 1 else None
                )
            elif variant == 4:
                effect = ("let" if family % 2 == 0 else "pm") if bar < 3 else None
            elif variant == 5:
                # Alternating muted and open attacks are a normal guitar riff,
                # and require choosing among several identical short labels.
                effect = "pm" if ei % 2 == 0 else None
            elif variant == 6:
                effect = "let" if ei in (0, count - 2) else None
            else:
                raise ValueError(f"Unknown technique study variant: {variant}")
            notes = []
            for string in picked:
                fret = shape[string - 1] + (
                    2 if string == 6 and tuning[-1] == 38 else 0
                )
                effects = [effect] if effect else []
                if variant == 3 and len(picked) > 1 and 0 < ei < count - 1:
                    effects = ["pm" if string == max(picked) else "let"]
                notes.append(
                    dict(
                        string=string,
                        fret=fret,
                        pitch=tuning[string - 1] + fret + 12,
                        effects=effects,
                    )
                )
            beat_duration = dict(
                value=duration,
                dotted=False,
                double_dotted=False,
                tuplet_enters=1,
                tuplet_times=1,
            )
            events.append(
                dict(
                    start=tick,
                    duration=beat_duration,
                    status="normal",
                    notes=notes,
                    effects=[f"chord:{chord}"] if ei == 0 else [],
                )
            )
            tick += int(duration_ticks(beat_duration))
        parsed = parse_measure_target("M | V0{@0:w:r}")
        parsed.update(
            time_signature=meter if bar == 0 else None,
            print_time_signature=bar == 0,
            voices=[dict(voice=0, events=events)],
        )
        if bar == 0 and mode != "tab":
            parsed.update(key_signature=key, print_key_signature=True)
        row = dict(
            id=f"{case}-{bar}",
            family=identity,
            source_id=case,
            split=split,
            source_measures=4,
            measure_index=bar,
            bar_index=bar,
            measure_number=bar + 1,
            part_id="part-1",
            staff_id="staff-1",
            part_name="Guitar",
            instrument="guitar",
            midi_program=24,
            mode=mode,
            tuning=tuning,
            capo=0,
            pitch_context=dict(
                clef="G2",
                clef_octave=0,
                instrument_transpose=-12,
                capo=0,
                octave_spans=[],
            ),
            visual_pitch=True,
            written_pitch=mode != "tab",
            pitch_reference="fingering" if mode == "tab" else "written",
            score_state=dict(time=meter, key=key_fifths(key), tuning=tuning),
            parsed=parsed,
            target=format_measure_target(parsed, mode),
            corpus="composed",
            renderer="alphatab",
            scenario=f"span-boundary-{variant}",
            composition=progression.split(),
            technique_semantics="visible event union; source note owners preserved",
        )
        errors = audit(row)
        if errors:
            raise ValueError((row["id"], errors))
        rows.append(row)
    return dict(
        id=case,
        title=title,
        mode=mode,
        measures=rows,
        render_options=dict(
            width=(1120, 1620)[layout],
            scale=(1.3, 1.6)[layout],
            bars_per_row=(2, 4)[layout],
            capture_techniques=True,
        ),
    )



def grounding_rows(state, geometry, page_path, root):
    rows = []
    with Image.open(page_path) as image:
        for mi, (row, box) in enumerate(zip(state['measures'], geometry['boxes'], strict=True)):
            x, y, w, h = box
            crop = [max(0, int(x)-3), max(0, int(y)-3),
                    min(image.width, int(x+w)+4), min(image.height, int(y+h)+4)]
            path = root / f'measure-{mi}.png'
            image.crop(crop).save(path)
            features, anchors = event_features(parse_measure_target(row['target']), row['mode'])
            drawn = {(a['voice'], a['onset']): a for a in geometry['eventAnchors'] if a['measure'] == mi}
            positions = []
            for voice, onset in anchors:
                views = drawn[voice, onset]['views']
                bounds = views[-1 if row['mode'] == 'both' else 0]['bbox']
                positions.append([(bounds[0]+bounds[2]/2-crop[0])/(crop[2]-crop[0]),
                                  (bounds[1]+bounds[3]/2-crop[1])/(crop[3]-crop[1])])
            rows.append(dict(id=row['id'], family=row['family'], split=row['split'], mode=row['mode'],
                             image=str(path), source_page=str(page_path), bbox=crop, target=row['target'],
                             anchors=anchors, features=features.tolist(), positions=positions))
    return rows


def build(output, splits=('train', 'validation'), variants=4, families=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifests = {kind: {split: [] for split in splits} for kind in ('grounding', 'relations')}
    with browser_page() as (page, origin):
        for family in range(len(COMPOSITIONS)) if families is None else families:
            split = 'train' if family < 12 else 'validation' if family < 16 else 'test'
            if split not in splits:
                continue
            for mode in ('tab', 'notation', 'both'):
                for variant in range(variants):
                    for layout in (0, 1):
                        state = recipe(family, mode, variant, layout)
                        root = output / split / state['id']
                        root.mkdir(parents=True)
                        page_path = root / 'page.png'
                        page.goto(origin + '/research/data/score_renderer.js')
                        geometry = page.evaluate("""async ({state, origin}) => {
                            const {render} = await import(origin + '/research/data/score_renderer.js');
                            return await render(state);
                        }""", dict(state=state, origin=origin))
                        page.locator('#score').screenshot(path=str(page_path))
                        (root / 'geometry.json').write_text(json.dumps(geometry))
                        (root / 'score.json').write_text(json.dumps(state))
                        manifests['grounding'][split].extend(grounding_rows(state, geometry, page_path, root))
                        rows = extract(state, geometry, page_path, root)
                        by_image = {r['image']: r for r in rows}
                        for row in rows:
                            row.update(split=split, family=state['measures'][0]['family'], variant=variant, layout=layout)
                            compiled = compile_row(row, previous_row=by_image.get(previous_image(row)))
                            manifests['relations'][split].append(compiled)
                            if continuation := continuation_variant(compiled, output):
                                manifests['relations'][split].append(continuation)
            print(f'{split}: rendered family {family}', flush=True)
    reports = {}
    for kind, subsets in manifests.items():
        folder = output / kind
        folder.mkdir()
        reports[kind] = {}
        for split, rows in subsets.items():
            (folder / f'manifest_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            reports[kind][split] = dict(images=len(rows), families=sorted({r['family'] for r in rows}))
        (folder / 'provenance.json').write_text(json.dumps(reports[kind], indent=2))
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    p.add_argument('--variants', type=int, default=4)
    p.add_argument('--families', nargs='+', type=int)
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
