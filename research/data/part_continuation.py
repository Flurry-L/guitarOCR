"""Prepare part continuation data from native GP exports and engraved scores."""

import argparse
import io
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from research.common.part_continuation import ROW_SIZE, contain, continuation_inputs
from research.common.pdf import open_pdf
from research.data.catalog import write_json, write_jsonl


def prepare_native_document(job):
    source, layout_path, output = job
    path = Path(layout_path)
    trace = json.loads(path.read_text())
    score = json.loads(path.with_name('official-score.json').read_text())
    native_staff = score['tracks'][0]['staves'][0]
    identity = f"native-{source['source_id']}"
    destination = Path(output) / identity
    destination.mkdir(parents=True, exist_ok=True)
    systems, rendered = [], {}
    with open_pdf(path.with_name('score.pdf')) as pdf:
        candidates = trace['systems']
        # Bound image preparation while retaining genuine page transitions and
        # larger musical changes within the same underlying track.
        indices = sorted(set([0, 1, len(candidates)-1] +
                             [i for i, s in enumerate(candidates) if i and s['page'] != candidates[i-1]['page']]))[:8]
        for index in indices:
            system = candidates[index]
            page = int(system['page'])
            if page not in rendered:
                rendered[page] = pdf[page-1].render(180).convert('L')
            image = rendered[page]
            boxes = [r['bbox_mm'] for r in system['measure_boxes']]
            y = min(b[1] for b in boxes) * 180 / 25.4
            bottom = max(b[1]+b[3] for b in boxes) * 180 / 25.4
            crop = image.crop((0, max(0, int(y)-12), image.width, min(image.height, int(bottom)+13)))
            pixels = contain(crop, ROW_SIZE)
            cache = destination / f'{page}-{index}.npz'
            np.savez_compressed(cache, visible=pixels, absent=pixels)
            systems.append(dict(page=page, system=index, rows=[dict(part='part-1', mode=source['mode'],
                instrument=source['instrument'], cache=str(cache.resolve()), source_family=source['family'],
                clef='tab' if source['mode'] == 'tab' else native_staff['measures'][system['first_measure_index']]['clef'],
                name=score['tracks'][0]['name'],
                label_present=bool(system.get('track_label')))]))
    return dict(id=identity, split=source['split'], source_families=[source['family']],
                source_track=source['source_id'], domain='native', systems=systems)


def render_engraved_document(job):
    import cairosvg
    import verovio

    source_path, output, font = job
    source = json.loads(source_path.read_text())
    root = output / source["id"]
    root.mkdir(parents=True, exist_ok=False)
    renderer = verovio.toolkit()
    renderer.setOptions(
        {
            "scale": 70,
            "pageWidth": 2100,
            "pageHeight": 2970,
            "svgBoundingBoxes": True,
            "header": "auto",
            "footer": "none",
            "font": font,
        }
    )
    if not renderer.loadFile(str(source_path.parent / "score.musicxml")):
        raise ValueError(f"Cannot render source: {source_path}")
    flat = [(part, staff) for part in source["parts"] for staff in part["staves"]]
    systems = []
    changed = []
    for page in range(1, renderer.getPageCount() + 1):
        svg = renderer.renderToSVG(page)
        tree = ET.fromstring(svg)
        parents = {child: parent for parent in tree.iter() for child in parent}
        visible = Image.open(
            io.BytesIO(
                cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
            )
        ).convert("L")
        nested = next(
            e
            for e in tree.iter()
            if e is not tree and e.tag.endswith("svg") and "viewBox" in e.attrib
        )
        scale = visible.width / float(nested.get("viewBox").split()[2])
        bounds = []
        for system_index, system in enumerate(
            n for n in tree.iter() if n.get("class") == "system"
        ):
            parts = defaultdict(list)
            for measure in (n for n in system if n.get("class") == "measure"):
                staves = [n for n in measure if n.get("class") == "staff"]
                if len(staves) != len(flat):
                    raise ValueError("Renderer changed the staff count")
                for staff_node, (part, staff) in zip(staves, flat, strict=True):
                    if staff["id"] != "staff-1":
                        continue
                    boxes = [
                        n
                        for n in staff_node.iter()
                        if n.tag.endswith("rect")
                        and float(n.get("width", 0)) > 0
                        and float(n.get("height", 0)) > 0
                    ]
                    if not boxes:
                        raise ValueError("Staff has no source bounding box")
                    tx = ty = 0.0
                    ancestor = staff_node
                    while ancestor in parents:
                        for x, y in re.findall(
                            r"translate\(\s*([-\d.]+)[, ]+([-\d.]+)\s*\)",
                            ancestor.get("transform", ""),
                        ):
                            tx += float(x)
                            ty += float(y)
                        ancestor = parents[ancestor]
                    top = (min(float(n.get("y")) for n in boxes) + ty) * scale
                    bottom = (
                        max(float(n.get("y")) + float(n.get("height")) for n in boxes)
                        + ty
                    ) * scale
                    parts[part["id"]].append((top, bottom, part))
            for part_id, members in parts.items():
                top = max(0, int(min(m[0] for m in members)) - 34)
                bottom = min(visible.height, int(max(m[1] for m in members)) + 35)
                bounds.append((system_index, part_id, top, bottom, members[0][2]))
        for node in list(tree.iter()):
            if node.get("class") in {"label", "labelAbbr"}:
                parents[node].remove(node)
        absent = Image.open(
            io.BytesIO(
                cairosvg.svg2png(bytestring=ET.tostring(tree), background_color="white")
            )
        ).convert("L")
        visible.save(root / f"page-{page}.png")
        grouped = defaultdict(list)
        for system, part_id, top, bottom, part in bounds:
            crop = (0, top, visible.width, bottom)
            a, b = (
                contain(visible.crop(crop), ROW_SIZE),
                contain(absent.crop(crop), ROW_SIZE),
            )
            count = int(np.count_nonzero(a != b))
            changed.append(count)
            path = root / f"{page}-{system}-{part_id}.npz"
            np.savez_compressed(path, visible=a, absent=b)
            grouped[system].append(
                {
                    "part": part_id,
                    "mode": "notation",
                    "instrument": part["instrument"],
                    "clef": part['staves'][0]['measures'][0]['pitch_context']['clef'],
                    "name": part['name'],
                    "cache": str(path.resolve()),
                    "source_family": source["id"],
                    "label_pixels": count,
                }
            )
        systems.extend(
            {"page": page, "system": system, "rows": rows}
            for system, rows in sorted(grouped.items())
        )
    if len(systems) < 2:
        raise ValueError("Continuation requires at least two native systems")
    return {
        "id": source["id"],
        "domain": "native_engraved",
        "split": source["split"],
        "source_families": [source["id"]],
        "font": font,
        "systems": systems,
        "label_rows": sum(n > 0 for n in changed),
        "rows": len(changed),
    }




@lru_cache(maxsize=8192)
def pixels(path, absent):
    with np.load(path) as cache:
        return cache['absent' if absent else 'visible'].copy()


def visible_label_variant(record, rng):
    """Training-only full/short labels from the renderer's paired label mask.

    The score pixels are shared by the visible and erased training renders.
    Remove only a suffix of their label difference; inference never receives
    the mask, its coordinates, or the rendered part identity.
    """
    absent = pixels(record['cache'], True).copy()
    style = rng.randrange(3)
    if style == 0:
        return absent
    visible = pixels(record['cache'], False).copy()
    if style == 1:
        return visible
    yy, xx = np.nonzero(visible != absent)
    if not len(xx):
        return visible
    boundary = round(int(xx.min()) + (int(xx.max()) - int(xx.min()) + 1) * rng.uniform(.4, .9))
    visible[:, boundary:] = absent[:, boundary:]
    return visible


def scale_row(row, scale, rng):
    width = max(1, round(row.shape[1] * scale))
    height = max(1, round(row.shape[0] * scale))
    resized = Image.fromarray(row).resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('L', ROW_SIZE, 255)
    canvas.paste(resized, (rng.randrange(ROW_SIZE[0] - width + 1),
                           rng.randrange(ROW_SIZE[1] - height + 1)))
    return np.asarray(canvas, dtype=np.uint8)


class ContinuationData(Dataset):
    def __init__(self, records, count, training, *, visible_labels=False):
        self.records = records
        self.count = count
        self.training = training
        self.visible_labels = visible_labels

    def __len__(self):
        return self.count

    def example(self, index, include_correspondence=False):
        rng = random.Random(390177 + index + (0 if self.training else 9000000))
        document = self.records[index % len(self.records)]
        first, last = sorted(rng.sample(range(len(document['systems'])), 2))
        left = list(document['systems'][first]['rows'])
        right = list(document['systems'][last]['rows'])
        prior = rng.choice(left)
        roll = rng.random()
        target = 1 if roll < .4 else 0 if roll < .8 else 2
        if document.get('domain') == 'native':
            following = document
            if target == 0:
                # Single-track PDF exports have one active part. Different
                # tracks are supervised in native_group galleries containing
                # every active candidate, rather than as unrelated documents.
                target = 1
            systems = [s for s in following['systems'][1:] if not s['rows'][0].get('label_present')]
            right = list(rng.choice(systems or following['systems'][1:])['rows'])
            current = right[0]
        elif target == 0:
            candidates = [r for r in right if r['part'] != prior['part']]
            same_instrument = [r for r in candidates if r['instrument'] == prior['instrument']]
            current = rng.choice(same_instrument if same_instrument and rng.random() < .75 else candidates)
        else:
            current = next(r for r in right if r['part'] == prior['part'])
        # Native score pages preserve the printed part order. Keep that real
        # convention in most examples. Changed galleries require visible names
        # or distinct staff profiles; hidden source IDs are not visual evidence.
        changed = len(left) > 1 and rng.random() < .3
        show_names = changed and rng.random() < .5 and all(
            np.any(pixels(r['cache'], False) != pixels(r['cache'], True)) for r in left + right)
        if changed:
            if not show_names:
                # A missing/added row makes the identity uncertainty visible.
                # Shuffling anonymous identical staff profiles at unchanged
                # counts would fabricate an unobservable convention break.
                if rng.random() < .5:
                    dropped = rng.choice([r for r in left if r is not prior])
                    left = [r for r in left if r is not dropped]
                else:
                    dropped = rng.choice([r for r in right if r is not current])
                    right = [r for r in right if r is not dropped]
            rng.shuffle(left)
            rng.shuffle(right)
        li, ri = left.index(prior), right.index(current)
        before = [pixels(r['cache'], rng.random() < .6).copy() for r in left]
        after = [pixels(r['cache'], True).copy() for r in right]
        if self.visible_labels:
            label_rng = random.Random(910397 + index + (0 if self.training else 8000000))
            after = [visible_label_variant(r, label_rng) for r in right]
        def full_label(row, image):
            visible, absent = pixels(row['cache'], False), pixels(row['cache'], True)
            return bool(np.any(visible != absent) and np.array_equal(image, visible))
        if show_names:
            # Explicit names make a genuine rearrangement identifiable. In the
            # remaining cases partial/absent labels preserve actual ambiguity.
            before = [pixels(r['cache'], False).copy() for r in left]
            after = [pixels(r['cache'], False).copy() for r in right]
        def signature(row):
            return row['mode'], row['clef']
        previous_counts = Counter(signature(r) for r in left)
        current_counts = Counter(signature(r) for r in right)
        previous_names = Counter(r['name'] for r in left)
        current_names = Counter(r['name'] for r in right)
        resolvable = []
        for c, row in enumerate(right):
            matching = next((p for p, previous in enumerate(left) if previous['part'] == row['part']), None)
            named = (full_label(row, after[c]) and current_names[row['name']] == 1
                     and (matching is not None and previous_names[row['name']] == 1
                          and full_label(left[matching], before[matching])
                          or matching is None and row['name'] not in previous_names
                          and all(full_label(r, image) for r, image in zip(left, before, strict=True))))
            distinct = (previous_counts[signature(row)] == 0 or matching is not None
                        and previous_counts[signature(row)] == current_counts[signature(row)] == 1)
            resolvable.append(not changed or named or distinct)
        if changed and not resolvable[ri] and signature(prior) == signature(current):
            target = 2
        duplicated = False
        if target == 2:
            if not changed or resolvable[ri]:
                # Repeated anonymous candidates carry no unique identity.
                after.append(after[ri].copy())
                resolvable[ri] = False
                resolvable.append(False)
                duplicated = True
                if rng.random() < .5:
                    ri = len(after) - 1
        current_rows = right + ([current] if duplicated else [])
        pairs = defaultdict(list)
        for c, row in enumerate(current_rows):
            for p, previous in enumerate(left):
                label = (2 if not resolvable[c] and signature(previous) == signature(row)
                         else int(previous['part'] == row['part']))
                pairs[label].append((p, c))
        # Supervise the whole gallery, including unaffected rows beside an
        # ambiguous duplicate. The query mask identifies which relation is
        # being classified; uncertainty does not apply to every row at once.
        target = rng.choice(sorted(pairs))
        li, ri = rng.choice(pairs[target])
        prior, current = left[li], current_rows[ri]
        if self.training:
            contrast = rng.uniform(.75, 1.1)
            before = [np.clip(255 - (255 - v.astype(np.float32)) * contrast, 0, 255).astype(np.uint8) for v in before]
            after = [np.clip(255 - (255 - v.astype(np.float32)) * contrast, 0, 255).astype(np.uint8) for v in after]
            def vary(row):
                if rng.random() > .6:
                    return row
                return scale_row(row, rng.uniform(.75, 1.0), rng)
            before = [vary(row) for row in before]
            after = [vary(row) for row in after]
        # Correspondence must also work before any printed identity is known.
        # The fixed validation/test sampling includes the same unknown case.
        previous_identity = None if rng.random() < .5 else prior['instrument']
        flags = np.asarray([target == 0 and prior['instrument'] == current['instrument'],
                            len(before) == len(after) == 1, li != ri,
                            document.get('domain', '').startswith('native'), previous_identity is None], dtype=np.int64)
        example = dict(previous=before, current=after, previous_index=li, current_index=ri,
            previous_identity=previous_identity, previous_mode=prior['mode'], current_mode=current['mode'],
            target=target, flags=flags)
        if include_correspondence:
            example['correspondence'] = dict(previous_parts=[r['part'] for r in left],
                current_parts=[r['part'] for r in current_rows],
                previous_identities=[r['instrument'] if previous_identity is not None else None for r in left],
                previous_modes=[r['mode'] for r in left], current_modes=[r['mode'] for r in current_rows],
                previous_instruments=[r['instrument'] for r in left],
                current_instruments=[r['instrument'] for r in current_rows],
                resolvable=resolvable, gallery_changed=changed or duplicated,
                domain=document.get('domain', 'ensemble'), family=document['source_families'])
        return example

    def __getitem__(self, index):
        example = self.example(index)
        target, flags = example.pop('target'), example.pop('flags')
        return (*continuation_inputs(**example), np.int64(target), flags)


def group_native_tracks(rows):
    """Build one candidate gallery from the tracks of the same source and mode."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_families'][0], row['systems'][0]['rows'][0]['mode']].append(row)
    grouped, members = [], set()
    for (family, mode), values in groups.items():
        tracks = {row['source_track']: row for row in values}
        if not 2 <= len(tracks) <= 4:
            continue
        members.update(row['id'] for row in tracks.values())
        systems = []
        for index in range(min(len(row['systems']) for row in tracks.values())):
            observed = [dict(deepcopy(row['systems'][index]['rows'][0]), part=track)
                        for track, row in tracks.items()]
            systems.append(dict(page=index + 1, system=index, rows=observed))
        grouped.append(dict(id=f'native-group-{family}-{mode}', domain='native_group',
                            split=values[0]['split'], source_families=[family], systems=systems))
    return [row for row in rows if row['id'] not in members] + grouped




def native_jobs(exports, split, output, excluded, limit, rng):
    jobs, seen = [], set()
    for export in exports:
        for line in (export / 'document-manifest.jsonl').read_text().splitlines():
            document = json.loads(line)
            document_split = {'dev': 'validation'}.get(document['split'], document['split'])
            if document_split != split or document['source_family_id'] in excluded:
                continue
            for path in sorted((export / 'documents' / document['document_id']).glob('tracks/*/layout.json')):
                source_id = f"{document['document_id']}-{path.parent.name}"
                if source_id in seen:
                    continue
                trace = json.loads(path.read_text())
                if len(trace['systems']) < 2 or not any(not s.get('track_label') for s in trace['systems'][1:]):
                    continue
                seen.add(source_id)
                source = dict(source_id=source_id, family=document['source_family_id'], split=split,
                              mode=document['display_mode'], instrument=document['instrument_kind'])
                jobs.append((source, str(path), str(output)))
    rng.shuffle(jobs)
    return jobs[:limit or None]


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    excluded = ({json.loads(line)['provenance']['family']
                 for line in args.exclude_families.read_text().splitlines()}
                if args.exclude_families else set())
    fonts = ['Bravura', 'Leipzig', 'Gootville', 'Leland', 'Petaluma']
    families, summary = {}, {}
    with ProcessPoolExecutor(args.workers) as pool:
        for split in ['train', 'validation', 'test']:
            rng = random.Random(args.seed)
            rows = []
            if args.native_exports:
                jobs = native_jobs(args.native_exports, split, args.output / split / 'native', excluded, args.limit, rng)
                rows.extend(group_native_tracks(list(pool.map(prepare_native_document, jobs))))
            if args.engraved:
                sources = []
                for path in sorted(args.engraved.glob('**/score.json')):
                    value = json.loads(path.read_text())
                    if value['split'] == split and len(value['parts']) >= 2 and value['id'] not in excluded:
                        sources.append(path)
                rng.shuffle(sources)
                jobs = [(path, args.output / split / 'engraved', fonts[i % len(fonts)])
                        for i, path in enumerate(sources[:args.limit or None])]
                rows.extend(pool.map(render_engraved_document, jobs))
            if any(row['split'] != split or len(row['systems']) < 2 for row in rows):
                raise ValueError('Continuation data requires matching source splits and at least two systems')
            if len({row['id'] for row in rows}) != len(rows):
                raise ValueError('The same document was supplied through multiple continuation sources')
            families[split] = {family for row in rows for family in row['source_families']}
            if any(families[split] & values for name, values in families.items() if name != split):
                raise ValueError('Continuation source families cross dataset splits')
            write_jsonl(args.output / f'{split}.jsonl', rows)
            summary[split] = dict(documents=len(rows), families=len(families[split]),
                                  domains=dict(Counter(row.get('domain', 'ensemble') for row in rows)))
            print(split, summary[split], flush=True)
    arguments = {key: str(value) if isinstance(value, Path) else
                 [str(p) for p in value] if isinstance(value, list) else value
                 for key, value in vars(args).items()}
    write_json(args.output / 'provenance.json', dict(arguments=arguments, splits=summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-exports', type=Path, nargs='+', help='Native export roots with document-manifest.jsonl and documents/')
    parser.add_argument('--engraved', type=Path, help='Engraved score directory with {split}/*/score.json')
    parser.add_argument('--exclude-families', type=Path, help='Optional information manifest whose families must be excluded')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Maximum documents per source and split; 0 uses all')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=202610039)
    args = parser.parse_args()
    if not any((args.native_exports, args.engraved)):
        parser.error('Provide --native-exports or --engraved')
    if args.limit < 0 or args.workers < 1:
        parser.error('limit must be nonnegative and workers must be positive')
    build(args)


if __name__ == '__main__':
    main()
