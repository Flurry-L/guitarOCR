"""Fresh playable compositions for metadata across real alphaTab score layouts."""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image
from scorelib.score import format_measure_target, parse_measure_target

from research.data.prompts import CLEF_PROMPT
from research.data.score_rendering import audit, browser_page
from research.data.training_samples import chat, staff_sample
from research.defaults import PROJECT_ROOT

PROFILES = [
    ("Guitar", "guitar", 24, [64, 59, 55, 50, 45, 40], "G2", -12),
    ("E. Guitar", "guitar", 27, [64, 59, 55, 50, 45, 40, 35], "G2", -12),
    ("Tenor Gtr", "guitar", 24, [69, 62, 55, 48], "G2", -12),
    ("8-str Gtr", "guitar", 27, [64, 59, 55, 50, 45, 40, 35, 30], "G2", -12),
    ("Bass", "bass", 33, [43, 38, 33, 28], "F4", -12),
    ("E. Bass", "bass", 34, [43, 38, 33, 28, 23], "F4", -12),
    ("6-str Bass", "bass", 34, [48, 43, 38, 33, 28, 23], "F4", -12),
    ("Violin", "pitched", 40, [], "G2", 0),
    ("Viola", "pitched", 41, [], "C3", 0),
]


def composition(split, index):
    rng = random.Random(
        {"train": 719100, "validation": 942700, "test": 1265300}[split] + index
    )
    name, instrument, program, tuning, clef, transpose = PROFILES[index % len(PROFILES)]
    mode = (
        ("tab", "both", "notation")[(index // len(PROFILES)) % 3]
        if tuning
        else "notation"
    )
    family = f"information-phrase-{split}-{index:04d}"
    rows = []
    for bi in range(8):
        row = dict(
            part_id='part-1', staff_id='staff-1', capo=0, visual_pitch=True,
            corpus='composed', renderer='alphatab', scenario='information',
            pitch_context=dict(clef=clef, clef_octave=0, instrument_transpose=transpose,
                               capo=0, octave_spans=[]),
            parsed=parse_measure_target('M | V0{@0:w:r}'),
        )
        row.update(
            id=f"{family}-{bi}",
            family=family,
            source_id=family,
            split=split,
            source_measures=8,
            measure_index=bi,
            bar_index=bi,
            measure_number=bi + 1,
            part_name=name,
            instrument=instrument,
            midi_program=program,
            tuning=tuning,
            mode=mode,
            pitch_reference="fingering" if mode == "tab" else "written",
            written_pitch=mode != "tab",
            score_state={"time": "4/4", "key": 0, "tuning": tuning},
        )
        row["pitch_context"].update(clef=clef, instrument_transpose=transpose)
        events = []
        # Independent diatonic/tablature phrases, with complete bar duration.
        for ei in range(8):
            if tuning:
                string = rng.randint(1, len(tuning))
                fret = rng.choice((0, 2, 3, 5, 7, 9))
                note = {
                    "pitch": tuning[string - 1] + fret - transpose,
                    "string": string,
                    "fret": fret,
                    "effects": [],
                }
            else:
                note = {
                    "pitch": (55 if clef == "C3" else 60)
                    + rng.choice((0, 2, 4, 5, 7, 9, 11, 12)),
                    "effects": [],
                }
            events.append(
                {
                    "start": ei * 480,
                    "duration": {
                        "value": 8,
                        "dotted": False,
                        "double_dotted": False,
                        "tuplet_enters": 1,
                        "tuplet_times": 1,
                    },
                    "status": "normal",
                    "notes": [note],
                    "effects": [],
                }
            )
        row["parsed"].update(
            time_signature="4/4" if bi == 0 else None,
            print_time_signature=bi == 0,
            key_signature="CMajor" if bi == 0 and mode != "tab" else None,
            print_key_signature=bi == 0 and mode != "tab",
            voices=[{"voice": 0, "events": events}],
        )
        row["target"] = format_measure_target(row["parsed"], mode)
        if errors := audit(row):
            raise ValueError((family, errors))
        rows.append(row)
    return {
        "id": family,
        "title": f"Study {rng.randrange(10000, 99999)}",
        "mode": mode,
        "measures": rows,
        "render_options": {
            "width": rng.choice((1050, 1400, 1800)),
            "scale": rng.choice((1.1, 1.4, 1.7)),
            "bars_per_row": 2,
            "capture_techniques": True,
        },
    }


def build(output, train=108, validation=27, test=36):
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    font = TTFont(PROJECT_ROOT / "ui/workbench/vendor/font/Bravura.woff2")
    glyphs = font.getGlyphSet()
    glyph_bounds = {}
    for code, name in font.getBestCmap().items():
        if 57424 <= code <= 57454:
            pen = BoundsPen(glyphs)
            glyphs[name].draw(pen)
            if pen.bounds:
                glyph_bounds[str(code)] = [
                    v / font["head"].unitsPerEm for v in pen.bounds
                ]

    output.mkdir(parents=True, exist_ok=True)
    data = {split: [] for split in ("train", "validation", "test")}
    relations = {split: [] for split in data}
    with browser_page() as (page, origin):
        for split, count in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]:
            for index in range(count):
                state = composition(split, index)
                folder = output / split / state["id"]
                folder.mkdir(parents=True, exist_ok=True)
                page.goto(origin + "/research/data/score_renderer.js")
                geometry = page.evaluate(
                    """async ({state, origin, glyphBounds}) => {
                    const {render} = await import(origin + '/research/data/score_renderer.js');
                    const geometry = await render(state);
                    const {alphaTab} = await import(origin + '/ui/workbench/score-engraving.js');
                    const names = Object.fromEntries(Object.entries(alphaTab.model.MusicFontSymbol)
                        .filter(([k,v]) => typeof v === 'number' && /Clef/.test(k)).map(([k,v]) => [v,k]));
                    const host = document.querySelector('#score').getBoundingClientRect();
                    const track = window.scoreRenderApi.score.tracks[0];
                    geometry.instrument_labels = [...document.querySelectorAll('#score svg text')]
                        .filter(n => n.textContent === track.shortName || n.textContent === track.name)
                        .map(n => { const b = n.getBoundingClientRect();
                            return {text:n.textContent,bbox:[b.x-host.x,b.y-host.y,b.width,b.height]}; });
                    geometry.clefs = [...document.querySelectorAll('#score svg text')].flatMap(n => {
                        const symbol = names[n.textContent.codePointAt(0)];
                        const bounds = glyphBounds[n.textContent.codePointAt(0)];
                        if (!symbol || !bounds) return [];
                        const css = getComputedStyle(n);
                        const [l,b,r,t] = bounds, size = parseFloat(css.fontSize);
                        const anchor = css.textAnchor, advance = n.getComputedTextLength();
                        const shift = anchor === 'middle' ? advance / 2 : anchor === 'end' ? advance : 0;
                        const x = Number(n.getAttribute('x') || 0) - shift;
                        const y = Number(n.getAttribute('y') || 0);
                        const matrix = n.getScreenCTM();
                        const corners = [[l,-t],[r,-t],[l,-b],[r,-b]].map(([gx,gy]) =>
                            new DOMPoint(x+gx*size,y+gy*size).matrixTransform(matrix));
                        const left = Math.min(...corners.map(p => p.x)) - host.x;
                        const top = Math.min(...corners.map(p => p.y)) - host.y;
                        const right = Math.max(...corners.map(p => p.x)) - host.x;
                        const bottom = Math.max(...corners.map(p => p.y)) - host.y;
                        return [{symbol, bbox:[left,top,right-left,bottom-top],
                            text_anchor:anchor, text_advance:advance, font_size:size,
                            outline_bounds_em:bounds, font_family:css.fontFamily,
                            coordinate_space:'host_screenshot_pixels'}];
                    });
                    return geometry;
                }""",
                    {"state": state, "origin": origin, "glyphBounds": glyph_bounds},
                )
                full = folder / "page.png"
                page.locator("#score").screenshot(path=str(full))
                first = state["measures"][0]
                if not geometry["instrument_labels"] or not geometry["clefs"]:
                    raise ValueError(
                        "Renderer did not expose visible instrument/clef evidence"
                    )
                visible_name = geometry["instrument_labels"][0]["text"]
                bands = [
                    [{"bbox": box} for box in geometry["boxes"][i : i + 2]]
                    for i in range(0, len(geometry["boxes"]), 2)
                ]
                provenance = {
                    "family": state["id"],
                    "split": split,
                    "renderer": "alphaTab",
                    "mode": first["mode"],
                }
                value = {
                    "parts": [
                        {
                            "name": visible_name,
                            "instrument": first["instrument"],
                            "strings": len(first["tuning"])
                            if first["mode"] != "notation"
                            else None,
                            "program": first["midi_program"],
                        }
                    ],
                    "rows": [[i, 0, 0] for i in range(len(bands))],
                }
                relations[split].append({
                    "image": str(full.resolve()), "expected": value,
                    "centers": [(min(r["bbox"][1] for r in band) +
                                 max(r["bbox"][1] + r["bbox"][3] for r in band)) / 2
                                for band in bands],
                    "row_modes": [first["mode"]] * len(bands),
                    "family": state["id"], "split": split,
                })
                with Image.open(full) as image:
                    top = min(b["bbox"][1] for b in bands[0])
                    bottom = max(b["bbox"][1] + b["bbox"][3] for b in bands[0])
                    lx, ly, lw, lh = geometry["instrument_labels"][0]["bbox"]
                    if not (
                        lx >= 0
                        and lx + lw <= image.width
                        and ly >= top - 12
                        and ly + lh <= bottom + 13
                    ):
                        raise ValueError(
                            f"Instrument label is not completely visible in staff crop: {state['id']}"
                        )
                    staff = folder / "staff.png"
                    image.crop(
                        (
                            0,
                            max(0, int(top) - 12),
                            image.width,
                            min(image.height, int(bottom) + 13),
                        )
                    ).save(staff)
                    data[split].append(
                        staff_sample(
                            staff,
                            {
                                "instrument": first["instrument"],
                                "string_count": value["parts"][0]["strings"],
                                "name": visible_name,
                            },
                            **provenance,
                            task="staff_styles",
                        )
                    )
                    for ci, glyph in enumerate(geometry["clefs"][:4]):
                        x, y, w, h = glyph["bbox"]
                        symbol = glyph["symbol"]
                        # The glyph identity is supplied by the renderer, not pixels.
                        clef = (
                            "tab"
                            if "Tab" in symbol
                            else "G2"
                            if symbol.startswith("GClef")
                            else "F4"
                            if symbol.startswith("FClef")
                            else first["pitch_context"]["clef"]
                            if symbol.startswith("CClef")
                            else "percussion"
                        )
                        octave = (
                            -12
                            if "8vb" in symbol
                            else 12
                            if "8va" in symbol
                            else -24
                            if "15mb" in symbol
                            else 24
                            if "15ma" in symbol
                            else 0
                        )
                        path = folder / f"clef-{ci}.png"
                        # Margin variation comes from source identity, never
                        # pixels. Keep the complete outline, including any
                        # integrated octave numeral, with nearby staff lines.
                        margin_rng = random.Random(f"{state['id']}:clef:{ci}:v4")
                        margins = [margin_rng.randint(3, 10) for _ in range(2)]
                        vertical = [
                            math.ceil(glyph["font_size"] * margin_rng.uniform(0.20, 0.35))
                            for _ in range(2)
                        ]
                        crop_bounds = (
                            max(0, math.floor(x) - margins[0]),
                            max(0, math.floor(y) - vertical[0]),
                            min(image.width, math.ceil(x + w) + margins[1]),
                            min(image.height, math.ceil(y + h) + vertical[1]),
                        )
                        if not (
                            crop_bounds[0] <= x < x + w <= crop_bounds[2]
                            and crop_bounds[1] <= y < y + h <= crop_bounds[3]
                        ):
                            raise ValueError(f"Incomplete clef outline: {state['id']}:{ci}")
                        glyph["crop_bounds"] = list(crop_bounds)
                        glyph["complete_outline_in_crop"] = True
                        image.crop(crop_bounds).save(path)
                        data[split].append(
                            chat(
                                [path],
                                CLEF_PROMPT,
                                {"clef": clef, "clef_octave": octave},
                                **provenance,
                                task="clef_styles",
                                renderer_symbol=symbol,
                                glyph_geometry=glyph,
                            )
                        )
                (folder / "score.json").write_text(json.dumps(state, indent=2))
                (folder / "geometry.json").write_text(
                    json.dumps(geometry, indent=2)
                )
                if index % 20 == 0:
                    print(split, index, count, flush=True)
    for split, rows in data.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (output / f"structure_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in relations[split])
        )
    print({split: len(rows) for split, rows in data.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=108)
    parser.add_argument("--validation", type=int, default=27)
    parser.add_argument("--test", type=int, default=36)
    build(**vars(parser.parse_args()))
