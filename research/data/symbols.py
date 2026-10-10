"""Build clef and signature crops from renderer-owned visible geometry."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image

from research.common.clef_spatial import LABELS, clef_context
from research.data.prompts import CLEF_PROMPT, SIGNATURE_PROMPT, signature_target
from research.data.training_samples import chat


def render_crop(xml, folder, font, dpi, element, padding, *, width=1300, element_index=0):
    import cairosvg
    import verovio

    renderer = verovio.toolkit()
    renderer.setOptions(dict(scale=50, pageWidth=width, pageHeight=700,
                             adjustPageHeight=True, breaks='none', font=font,
                             svgBoundingBoxes=True, header='none', footer='none', spacingStaff=12))
    if not renderer.loadData(xml):
        raise ValueError('Verovio rejected symbol composition')
    svg = renderer.renderToSVG(1)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'score.musicxml').write_text(xml)
    (folder / f'{font}.svg').write_text(svg)
    root = ET.fromstring(svg)
    node = [n for n in root.iter() if n.get('class') == element][element_index]
    boxes = [n for n in node.iter() if n.tag.endswith('rect')
             and float(n.get('width', 0)) > 0 and float(n.get('height', 0)) > 0]
    if not boxes:
        raise ValueError(f'Missing renderer bounds: {folder}')
    x = min(float(n.get('x')) for n in boxes)
    y = min(float(n.get('y')) for n in boxes)
    w = max(float(n.get('x')) + float(n.get('width')) for n in boxes) - x
    h = max(float(n.get('y')) + float(n.get('height')) for n in boxes) - y
    parents = {child: parent for parent in root.iter() for child in parent}
    ancestor = node
    while ancestor in parents:
        for dx, dy in re.findall(r'translate\(\s*([-\d.]+)[, ]+([-\d.]+)\s*\)', ancestor.get('transform', '')):
            x += float(dx)
            y += float(dy)
        ancestor = parents[ancestor]
    nested = next(n for n in root.iter() if n is not root and n.tag.endswith('svg') and 'viewBox' in n.attrib)
    png = cairosvg.svg2png(bytestring=svg.encode(), scale=dpi / 96, background_color='white')
    path = folder / f'{font}-{dpi}.png'
    with Image.open(BytesIO(png)) as image:
        scale = image.width / float(nested.get('viewBox').split()[2])
        pad = padding * dpi / 25.4
        if element == 'clef':
            query = [x * scale - pad, y * scale - pad, w * scale + 2 * pad, h * scale + 2 * pad]
            context, query = clef_context(image, query)
            context.convert('RGB').save(path)
            return path, query
        image.crop((max(0, int(x * scale - pad)), max(0, int(y * scale - pad)),
                    min(image.width, int((x + w) * scale + pad + 1)),
                    min(image.height, int((y + h) * scale + pad + 1)))).convert('RGB').save(path)
    return path


def render_clef(job):
    output, split, index, seed, fonts = job
    rng = random.Random(seed)
    label = LABELS[index % len(LABELS)]
    clef, octave = label["clef"], label["clef_octave"]
    sign = {"tab": "TAB", "percussion": "percussion"}.get(clef, clef[:1])
    line = int(clef[-1]) if clef[-1].isdigit() else 5 if clef == "tab" else 3
    count = rng.choice((4, 5, 6, 7, 8)) if clef == "tab" else 5
    change = (
        f"<clef-octave-change>{octave // 12}</clef-octave-change>" if octave else ""
    )
    notes = []
    for _ in range(8):
        pitch = rng.randrange(7)
        if clef == "percussion":
            body = "<unpitched><display-step>C</display-step><display-octave>5</display-octave></unpitched>"
        else:
            body = f"<pitch><step>{'CDEFGAB'[pitch]}</step><octave>{rng.choice((3, 4))}</octave></pitch>"
        technical = (
            f"<notations><technical><string>{rng.randint(1, count)}</string><fret>{rng.randint(0, 9)}</fret></technical></notations>"
            if clef == "tab"
            else ""
        )
        notes.append(
            f"<note>{body}<duration>1</duration><type>quarter</type>{technical}</note>"
        )
    xml = f"""<score-partwise version="4.0"><part-list><score-part id="P1"><part-name></part-name></score-part></part-list><part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>{rng.randint(-3, 3) if clef not in ("tab", "percussion") else 0}</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>{sign}</sign><line>{line}</line>{change}</clef><staff-details><staff-lines>{count}</staff-lines></staff-details></attributes>{"".join(notes[:4])}</measure><measure number="2">{"".join(notes[4:])}</measure></part></score-partwise>"""
    # Several genuinely rendered staves teach the query mask to select a
    # clef. Context alone must not make the topmost neighbouring sign win.
    tree = ET.fromstring(xml)
    part_list = tree.find('part-list')
    target_part = tree.find('part')
    tree.remove(target_part)
    part_list.clear()
    clefs = [rng.choice(LABELS) for _ in range(rng.choice((1, 2, 3)))]
    query_index = rng.randrange(len(clefs))
    clefs[query_index] = label
    from copy import deepcopy
    for part_index, part_label in enumerate(clefs):
        identity = f'P{part_index + 1}'
        listed = ET.SubElement(part_list, 'score-part', id=identity)
        ET.SubElement(listed, 'part-name').text = ''
        part = deepcopy(target_part)
        part.set('id', identity)
        attributes = part.find('./measure/attributes')
        rendered_clef = attributes.find('clef')
        rendered_clef.clear()
        value = part_label['clef']
        ET.SubElement(rendered_clef, 'sign').text = {'tab': 'TAB', 'percussion': 'percussion'}.get(value, value[:1])
        ET.SubElement(rendered_clef, 'line').text = value[-1] if value[-1].isdigit() else '5' if value == 'tab' else '3'
        if part_label['clef_octave']:
            ET.SubElement(rendered_clef, 'clef-octave-change').text = str(part_label['clef_octave'] // 12)
        attributes.find('staff-details/staff-lines').text = str(count if value == 'tab' else 5)
        tree.append(part)
    xml = ET.tostring(tree, encoding='unicode')
    font = fonts[(index // len(LABELS)) % len(fonts)]
    family = f"information-clef-{seed}"
    folder = Path(output) / "clefs" / split / family
    dpi = rng.choice((120, 150, 180, 240))
    padding = rng.uniform(0, .8)
    target, query = render_crop(xml, folder, font, dpi, 'clef', padding, element_index=query_index)
    return dict(chat([target], CLEF_PROMPT, label, family=family, split=split,
                     renderer="Verovio", task="clef", font=font), query_bbox=query)


def render_signature(job):
    output, index, numerator, denominator, key, font, dpi = job
    rng = random.Random(2026100300 + index)
    notes = []
    # A full-measure rest is conventional in every meter. The following bar
    # contains a complete diatonic phrase in the same meter, grouped by 2/3
    # for readable odd meters; no second contradictory time signature appears.
    kinds = {
        1: "whole",
        2: "half",
        4: "quarter",
        8: "eighth",
        16: "16th",
        32: "32nd",
        64: "64th",
    }
    remaining = numerator
    groups = []
    while remaining:
        count = 3 if remaining >= 3 and remaining != 4 else min(2, remaining)
        groups.append(count)
        remaining -= count
    pitch_index = rng.randrange(7)
    for group in groups:
        for position in range(group):
            pitch_index = max(0, min(13, pitch_index + rng.choice([-1, 0, 1, 2])))
            step = "CDEFGAB"[pitch_index % 7]
            octave = 4 + pitch_index // 7
            beam = ""
            if denominator >= 8 and group > 1:
                beam = f'<beam number="1">{"begin" if position == 0 else "end" if position == group - 1 else "continue"}</beam>'
            notes.append(
                f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>{256 // denominator}</duration><type>{kinds[denominator]}</type>{beam}</note>"
            )
    xml = f"""<?xml version="1.0" encoding="utf-8"?><score-partwise version="4.0"><part-list><score-part id="P1"><part-name></part-name></score-part></part-list><part id="P1"><measure number="1"><attributes><divisions>64</divisions><key><fifths>{key}</fifths></key><time><beats>{numerator}</beats><beat-type>{denominator}</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes><note><rest measure="yes"/><duration>{numerator * 256 // denominator}</duration></note></measure><measure number="2">{"".join(notes)}</measure></part></score-partwise>"""
    family = f"signature-{index}"
    path = render_crop(xml, Path(output) / "images" / family, font, dpi, 'staff', 2.5, width=1600)
    return {
        "messages": [{"role": "user", "content": "<image>" + SIGNATURE_PROMPT},
                     {"role": "assistant", "content": signature_target(f"{numerator}/{denominator}", key or None)}],
        "images": [str(path.resolve())], "source_id": family,
        "provenance": dict(family=family, split="train", renderer="Verovio", task="signature", font=font, dpi=dpi),
    }


def native_clef_crops(export, output):
    from research.common.pdf import open_pdf
    from research.data.native_supervision import visible_clef

    rows = []
    for line in (export / 'document-manifest.jsonl').read_text().splitlines():
        source = json.loads(line)
        split = {'dev': 'validation'}.get(source['split'], source['split'])
        for layout_path in sorted((export / 'documents' / source['document_id']).glob('tracks/*/layout.json')):
            layout = json.loads(layout_path.read_text())
            score = json.loads(layout_path.with_name('official-score.json').read_text())
            bars = {(staff_index, bar['measure_index']): bar
                    for staff_index, staff in enumerate(score['tracks'][0]['staves'])
                    for bar in staff['measures']}
            folder = output / 'guitarpro' / source['document_id'] / layout_path.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            with open_pdf(layout_path.with_name('score.pdf')) as document:
                rendered = {}
                for index, region in enumerate(layout['pitch_regions']):
                    if region['kind'] != 'clef':
                        continue
                    page = document[region['page'] - 1]
                    label = visible_clef(page, region, bars[region['staff_index'], region['measure_index']])
                    for variant, (dpi, padding) in enumerate(((120, .15), (180, .4), (240, 2.5))):
                        key = region['page'], dpi
                        if key not in rendered:
                            rendered[key] = page.render(dpi)
                        image = rendered[key]
                        x, y, width, height = region['bbox_mm']
                        scale = dpi / 25.4
                        query = [(x - padding) * scale, (y - padding) * scale,
                                 (width + 2 * padding) * scale, (height + 2 * padding) * scale]
                        context, query = clef_context(image, query)
                        path = folder / f'{index}-{variant}.png'
                        context.save(path)
                        rows.append(dict(chat([path], CLEF_PROMPT, label,
                                         family=source['source_family_id'], split=split,
                                         renderer='GuitarPro', task='clef', mode=source['display_mode'],
                                         source_document=source['document_id']), query_bbox=query))
    return rows


def engraved_clef_crops(dataset, output):
    """Use actual alphaTab clef draw bounds and their observed local labels."""
    rows = []
    for split in ('train', 'validation', 'test'):
        documents = {}
        for line in (dataset / f'manifest_{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            index = int(Path(row['image']).stem.removeprefix('measure-'))
            documents.setdefault(row['source_page'], {})[index] = row
        for source, measures in documents.items():
            page = Image.open(source).convert('RGB')
            geometry = json.loads(Path(source).with_name('geometry.json').read_text())
            for mark_index, mark in enumerate(geometry['painted']):
                if not any(0xE050 <= symbol <= 0xE06F for symbol in mark.get('symbols', [])):
                    continue
                measure = measures[mark['mi']]
                state = measure['observation']['state']
                if state.get('clef', {}).get('status') != 'observed':
                    continue
                label = dict(clef=state['clef']['value'], clef_octave=state['clef_octave']['value'])
                folder = output / 'alphatab' / measure['family']
                folder.mkdir(parents=True, exist_ok=True)
                for variant, padding in enumerate((0., .1, -.05)):
                    x, y, w, h = mark['bbox']
                    pad = max(w, h) * padding
                    context, query = clef_context(page, [x-pad, y-pad, w+2*pad, h+2*pad])
                    path = folder / f'{mark_index}-{variant}.png'
                    context.save(path)
                    rows.append(dict(chat([path], CLEF_PROMPT, label, family=measure['family'],
                        split=split, renderer='alphaTab', task='clef'), query_bbox=query))
    return rows


def build_clefs(output, train=2800, validation=560, test=560, seed=480100000,
          fonts=('Bravura', 'Leipzig', 'Gootville'), workers=6, native_exports=(), engraved=()):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    counts, offset = {}, 0
    with ProcessPoolExecutor(workers) as pool:
        for split, count in [('train', train), ('validation', validation), ('test', test)]:
            jobs = ((output, split, i, seed + offset + i, fonts) for i in range(count))
            with (output / f'{split}.jsonl').open('w') as stream:
                for row in pool.map(render_clef, jobs):
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[split] = count
            offset += count
    families = {}
    for export in native_exports:
        for row in native_clef_crops(export, output):
            family, split = (row['provenance'][key] for key in ('family', 'split'))
            if split not in counts or families.setdefault(family, split) != split:
                raise ValueError('Native clef families must retain their dataset split')
            with (output / f'{split}.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[split] += 1
    for dataset in engraved:
        for row in engraved_clef_crops(dataset, output):
            family, split = (row['provenance'][key] for key in ('family', 'split'))
            if families.setdefault(family, split) != split:
                raise ValueError('Engraved clef families must retain their dataset split')
            with (output / f'{split}.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[split] += 1
    return counts


def build_signatures(output, workers=8):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for numerator in range(1, 33):
        for di, denominator in enumerate((1, 2, 4, 8, 16, 32, 64)):
            for variant in range(3):
                index = ((numerator - 1) * 7 + di) * 3 + variant
                for font in ('Leipzig', 'Bravura'):
                    jobs.append((output, index, numerator, denominator, (index * 7) % 15 - 7,
                                 font, (150, 200, 300)[variant]))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(render_signature, jobs))
    for split in ('train', 'validation', 'test'):
        values = rows if split == 'train' else []
        (output / f'state_{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in values))
    return {'train': len(rows), 'validation': 0, 'test': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    clefs = commands.add_parser('clef', help='Source-disjoint clef crops')
    signatures = commands.add_parser('signature', help='Train-only complete meter coverage')
    for command in (clefs, signatures):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--workers', type=int, default=6)
    clefs.add_argument('--train', type=int, default=2800)
    clefs.add_argument('--validation', type=int, default=560)
    clefs.add_argument('--test', type=int, default=560)
    clefs.add_argument('--seed', type=int, default=480100000)
    clefs.add_argument('--fonts', nargs='+', default=['Bravura', 'Leipzig', 'Gootville'])
    clefs.add_argument('--native-exports', type=Path, nargs='+', default=[],
                       help='Native export roots with document-manifest.jsonl and documents/')
    clefs.add_argument('--engraved', type=Path, nargs='+', default=[],
                       help='Visible observation datasets with full page images and painted clef geometry')
    args = vars(parser.parse_args())
    build = build_clefs if args.pop('kind') == 'clef' else build_signatures
    print(json.dumps(build(**args), indent=2))


if __name__ == '__main__':
    main()
