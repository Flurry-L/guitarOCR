"""Read score systems and keep a stable part/staff timeline across pages."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from shared.instruments import program_from_visible_name


STRUCTURE_PROMPT = (
    'Read the score structure. Each staff row is marked R1, R2, ... in blue. '
    'A notation+TAB pair is ONE row of the same part. A piano grand staff has '
    'TWO rows of the same part, staff 0 for right hand and staff 1 for left hand. '
    'Group simultaneous parts into systems using brackets, barlines and names. '
    'Return JSON {"parts":[{"name":printed instrument name,"instrument":guitar|bass|pitched|drums,'
    '"strings":TAB line count or null,"program":zero-based GM program}],'
    '"rows":[[system,part,staff],...]}. Indices start at 0; rows follow the blue numbers. '
    'Keep the same part index in successive systems. Do not infer TAB strings from notation lines. '
)

STRUCTURE_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['parts', 'rows'],
    'properties': {
        'parts': {'type': 'array', 'minItems': 1, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'instrument', 'strings', 'program'],
            'properties': {
                'name': {'type': 'string'},
                'instrument': {'enum': ['guitar', 'bass', 'pitched', 'drums']},
                'strings': {'anyOf': [{'type': 'null'}, {'type': 'integer', 'minimum': 1, 'maximum': 12}]},
                'program': {'type': 'integer', 'minimum': 0, 'maximum': 127},
            },
        }},
        'rows': {'type': 'array', 'minItems': 1, 'items': {
            'type': 'array', 'minItems': 3, 'maxItems': 3,
            'items': {'type': 'integer', 'minimum': 0},
        }},
    },
}


def structure_schema(row_count):
    schema = deepcopy(STRUCTURE_SCHEMA)
    schema['properties']['rows'].update(minItems=row_count, maxItems=row_count)
    schema['properties']['parts']['maxItems'] = row_count
    return schema


def staff_rows(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['page'], row.get('row_index', row['system_index']))].append(row)
    return [sorted(values, key=lambda r: r['bbox'][0]) for _, values in sorted(groups.items())]


def structure_image(page, rows, output):
    with Image.open(page) as image:
        image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default(size=24)
    for i, row in enumerate(rows, 1):
        top = min(r['bbox'][1] for r in row)
        bottom = max(r['bbox'][1] + r['bbox'][3] for r in row)
        x = image.width - 62
        y = round((top + bottom) / 2) - 14
        draw.rectangle((x - 2, y - 2, image.width - 1, y + 30), fill='white')
        draw.text((x, y), f'R{i}', font=font, fill=(0, 65, 220))
    image.thumbnail((1200, 1680), Image.Resampling.LANCZOS)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return str(output.resolve())


def structure_model_image(marked, output=None):
    """Keep staff identities and both margins at their original resolution."""
    marked = Path(marked)
    output = Path(output) if output else marked.with_name(marked.stem + '.model.png')
    if output.exists() and output.stat().st_mtime_ns >= marked.stat().st_mtime_ns:
        return str(output.resolve())
    with Image.open(marked) as image:
        image = image.convert('RGB')
        left, right, gap = round(image.width * .32), round(image.width * .08), 24
        compact = Image.new('RGB', (left + right + gap, image.height), 'white')
        compact.paste(image.crop((0, 0, left, image.height)), (0, 0))
        compact.paste(image.crop((image.width - right, 0, image.width, image.height)), (left + gap, 0))
        output.parent.mkdir(parents=True, exist_ok=True)
        compact.save(output)
    return str(output.resolve())


def marked_systems(path, count):
    """Read continuous system barlines, returning None when the raster is unclear."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB')).astype(np.float32)
    blue = (rgb[:, :, 2] - rgb[:, :, 0] > 40) & (rgb[:, :, 2] - rgb[:, :, 1] > 20)
    ys = np.flatnonzero(blue[:, int(rgb.shape[1] * .9):].sum(1) >= 1)
    chunks = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1)
    centers = [float((c[0] + c[-1]) / 2) for c in chunks if len(c) >= 2]
    if len(centers) != count:
        return None
    ink = rgb.mean(2) < 210
    strength = ink[:, int(rgb.shape[1] * .22):int(rgb.shape[1] * .91)].mean(1)
    lines = np.flatnonzero(strength > .45)
    # A short final system may occupy less than half the page width. Its
    # horizontal lines are still dense inside a local window.
    width = max(8, round(rgb.shape[1] * .12))
    local = ink[:, int(rgb.shape[1] * .07):int(rgb.shape[1] * .91)]
    summed = np.pad(local.cumsum(1, dtype=np.int32), ((0, 0), (1, 0)))
    short_lines = np.flatnonzero(((summed[:, width:] - summed[:, :-width]).max(1) / width) > .9)
    staff = []
    for i, center in enumerate(centers):
        low = (centers[i - 1] + center) / 2 if i else 0
        high = (centers[i + 1] + center) / 2 if i + 1 < count else rgb.shape[0]
        hits = lines[(lines > low) & (lines < high)]
        if len(hits) < 4:
            hits = short_lines[(short_lines > low) & (short_lines < high)]
        if len(hits) < 4:
            return None
        staff.append(float(np.median(hits)))
    systems = [0]
    for first, second in zip(staff, staff[1:]):
        region = ink[round(first):round(second), :int(rgb.shape[1] * .23)]
        if not region.size:
            return None
        score = float(region.mean(0).max())
        if .7 < score < .9:
            return None
        systems.append(systems[-1] + int(score < .9))
    return systems


def constrain_rows(value, systems):
    """Use a repeated staff template only when it fits every visible system."""
    groups = defaultdict(list)
    for i, system in enumerate(systems):
        groups[system].append(i)
    widths = {len(rows) for rows in groups.values()}
    if len(widths) != 1:
        return
    width = widths.pop()
    parts, rows = value['parts'], value['rows']
    # Some generations repeat the same profile for successive systems.
    # Collapse only an exact periodic profile list compatible with the image.
    identities = [(re.sub(r'[^\w]', '', p['name'].casefold()), p['instrument'], p['strings'], p['program']) for p in parts]
    if len(parts) > width:
        period = next((n for n in range(1, width + 1) if all(v == identities[i % n] for i, v in enumerate(identities))), None)
        if period is None:
            return
        parts = parts[:period]
    template = []
    if rows and all(isinstance(r, list) and len(r) == 3 and all(type(v) is int for v in r) for r in rows):
        template = [(part, staff) for system, part, staff in rows if system == rows[0][0]]
    grand = {i for i, part in enumerate(parts) if re.search(r'\b(?:piano|keyboard|organ|harp)\b', part['name'], re.I)}
    grand_template = [(i, staff) for i in range(len(parts)) for staff in range(2 if i in grand else 1)]
    if grand and len(grand_template) == width:
        template = grand_template
    elif (len(template) != width or len(set(template)) != width
            or {p for p, _s in template} != set(range(len(parts)))
            or any(not 0 <= staff < 16 for _p, staff in template)):
        template = [(i, 0) for i in range(len(parts))]
        if len(template) != width:
            template = grand_template
    if len(template) != width:
        return
    value['parts'] = parts
    value['rows'] = [[system, part, staff] for system in groups for part, staff in template]


def parse_structure(raw, count, systems=None):
    value = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
    parts, rows = value['parts'], value['rows']
    if not isinstance(parts, list) or not parts or not isinstance(rows, list):
        raise ValueError('Structure needs parts and staff rows')
    for part in parts:
        if part.get('instrument') not in {'guitar', 'bass', 'pitched', 'drums'}:
            raise ValueError('Invalid part instrument')
        if part.get('strings') is not None and not 1 <= int(part['strings']) <= 12:
            raise ValueError('Invalid string count')
        if part.get('strings') is not None:
            part['strings'] = int(part['strings'])
        part['name'] = str(part.get('name') or part['instrument'])[:160]
        if not 0 <= int(part['program']) <= 127:
            raise ValueError('Invalid MIDI program')
        part['program'] = int(part['program'])
        visible_program = program_from_visible_name(part['name'])
        if visible_program is not None:
            part['program'] = visible_program
        elif part['instrument'] == 'guitar' and not 24 <= part['program'] <= 31:
            part['program'] = 25
        elif part['instrument'] == 'bass' and not 32 <= part['program'] <= 39:
            part['program'] = 33
    if systems is not None and len(systems) == count:
        constrain_rows(value, systems)
        parts, rows = value['parts'], value['rows']
    if len(rows) != count:
        raise ValueError('Structure does not cover every detected staff row')
    if systems is not None and len(systems) == count:
        actual = [r[0] for r in rows]
        order = {}
        actual = [order.setdefault(s, len(order)) for s in actual]
        if actual != systems:
            raise ValueError(f'Visible barlines group rows into systems {systems}; each connected system needs all its parts and staves')
    previous = -1
    seen = set()
    for system, part, staff in rows:
        if any(type(x) is not int for x in (system, part, staff)):
            raise ValueError('Structure indices must be integers')
        if system < 0 or system < previous or not 0 <= part < len(parts) or not 0 <= staff < 16:
            raise ValueError('Invalid system or part ordering')
        if (system, part, staff) in seen:
            raise ValueError('Repeated staff in the same system')
        seen.add((system, part, staff))
        previous = system
    return value


def aligned_columns(boxes, reference):
    """Match incomplete staff rows to distinct columns in reading order."""
    n, m = len(boxes), len(reference)
    if n == m:
        return list(range(n))
    costs = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    costs[0] = [0.] * (m + 1)
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i, row in enumerate(boxes, 1):
        centre = row['bbox'][0] + row['bbox'][2] / 2
        for j in range(i, m + 1):
            x, _y, width, _height = reference[j - 1]['bbox']
            matched = costs[i - 1][j - 1] + abs(centre - x - width / 2) / max(1, width)
            take[i][j] = matched <= costs[i][j - 1]
            costs[i][j] = min(matched, costs[i][j - 1])
    result, i, j = [], n, m
    while i:
        if take[i][j]:
            result.append(j - 1)
            i -= 1
        j -= 1
    return result[::-1]


def page_template(structure, rows):
    systems, order = defaultdict(list), {}
    for (system, part, staff), boxes in zip(structure['rows'], rows, strict=True):
        slot = order.setdefault(part, len(order))
        mode = Counter(r['mode'] for r in boxes).most_common(1)[0][0]
        systems[system].append((slot, staff, mode))
    signatures = {tuple(v) for v in systems.values()}
    return (signatures.pop() if len(signatures) == 1 else None), list(order)


def reconcile_page_profiles(pages):
    """Resolve isolated profile errors in repeated, complete staff templates.

    Condensed systems with missing parts have a different template and do not
    vote against complete systems. A strict page majority is required.
    """
    templates = defaultdict(list)
    for structure, rows in pages:
        template, order = page_template(structure, rows)
        if template is not None:
            templates[template].append((structure, order))
    for pages in templates.values():
        if len(pages) < 3:
            continue
        for slot in range(len(pages[0][1])):
            candidates = [s['parts'][ids[slot]] for s, ids in pages]
            keys = [(re.sub(r'[^\w]', '', p['name'].casefold()), p['instrument'],
                     p['strings'], p['program']) for p in candidates]
            winner, count = Counter(keys).most_common(1)[0]
            if count * 2 <= len(pages):
                continue
            profile = candidates[keys.index(winner)]
            for (structure, indices), key in zip(pages, keys, strict=True):
                if key != winner:
                    structure.setdefault('model_parts', deepcopy(structure['parts']))
                    structure['parts'][indices[slot]] = deepcopy(profile)


def complete_from_reference(raw, rows, systems, references):
    """Recover a missing part on a short page from its other complete systems."""
    if systems is None:
        return None
    try:
        incomplete = parse_structure(raw, len(rows))
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    groups = defaultdict(list)
    for system, boxes in zip(systems, rows, strict=True):
        groups[system].append(Counter(r['mode'] for r in boxes).most_common(1)[0][0])
    modes = {tuple(v) for v in groups.values()}
    if len(modes) != 1:
        return None
    mode = modes.pop()

    def identity(part):
        return (re.sub(r'[^\w]', '', part['name'].casefold()), part['instrument'], part['strings'], part['program'])

    observed = {identity(p) for p in incomplete['parts']}
    candidates = {}
    for page, structure, reference_rows in references:
        template, order = page_template(structure, reference_rows)
        if template is None or tuple(r[2] for r in template) != mode:
            continue
        profiles = [structure['parts'][i] for i in order]
        if len(profiles) <= len(incomplete['parts']) or not observed.issubset({identity(p) for p in profiles}):
            continue
        key = (tuple(identity(p) for p in profiles), template)
        candidates[key] = {'parts': deepcopy(profiles), 'rows': [
            [system, part, staff] for system in groups for part, staff, _mode in template], 'reference_page': page}
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return None


def read_structure(source, output, backend, cancelled=None, *, compact=False):
    from shared.tasks import Cancelled

    structured = getattr(backend, 'supports_json_schema', False)
    grouped = defaultdict(list)
    for row in staff_rows(source['records']):
        grouped[row[0]['page']].append(row)
    requests = []
    for page_number, rows in sorted(grouped.items()):
        path = structure_image(rows[0][0]['source_page'], rows, Path(output) / f'structure-{page_number}.png')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'url': structure_model_image(path) if compact else path},
            {'type': 'text', 'text': STRUCTURE_PROMPT + f'There are {len(rows)} marked rows.'}]}]
        requests.append((page_number, path, messages))
    initial = {}
    for offset in range(0, len(requests), 4):
        if cancelled and cancelled():
            raise Cancelled('已取消总谱结构识别')
        batch = requests[offset:offset + 4]
        messages = [r[2] for r in batch]
        generation = {'json_schema': [structure_schema(len(grouped[r[0]])) for r in batch]} if structured else {}
        outputs = backend.generate_batch(messages, 2048, **generation) if hasattr(backend, 'generate_batch') else [backend.generate(m, 2048, **generation) for m in messages]
        for request, (raw, _tokens) in zip(batch, outputs, strict=True):
            initial[request[0]] = (*request[1:], raw)
    geometry = {page: marked_systems(initial[page][0], len(rows)) for page, rows in grouped.items()}
    parsed, resolved_inputs = {}, {}
    for page_number, rows in sorted(grouped.items()):
        if cancelled and cancelled():
            raise Cancelled('已取消总谱结构识别')
        path, messages, raw = initial[page_number]
        systems_from_image = geometry[page_number]
        error = None
        generation = {'json_schema': structure_schema(len(rows))} if structured else {}
        for _attempt in range(2):
            if _attempt:
                raw, _ = backend.generate(messages, 2048, **generation)
            try:
                structure = parse_structure(raw, len(rows), systems_from_image)
                break
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                references = []
                for other, reference_rows in grouped.items():
                    if other == page_number:
                        continue
                    try:
                        reference = parsed.get(other) or parse_structure(initial[other][2], len(reference_rows), geometry[other])
                    except (ValueError, KeyError, TypeError, IndexError):
                        continue
                    references.append((other, reference, reference_rows))
                structure = complete_from_reference(raw, rows, systems_from_image, references)
                if structure is not None:
                    break
                messages.extend([{'role': 'assistant', 'content': [{'type': 'text', 'text': raw}]},
                                 {'role': 'user', 'content': [{'type': 'text', 'text': error + '. Return the complete corrected JSON.'}]}])
        else:
            raise ValueError(f'Cannot resolve score structure on page {page_number}: {error}')
        parsed[page_number] = structure
        resolved_inputs[page_number] = (path, raw, systems_from_image)
    reconcile_page_profiles([(parsed[page], rows) for page, rows in sorted(grouped.items())])
    predictions, parts, by_name, identities = [], [], {}, {}
    bar_offset = 0
    for page_number, rows in sorted(grouped.items()):
        structure = parsed[page_number]
        path, raw, systems_from_image = resolved_inputs[page_number]
        template, order = page_template(structure, rows)
        mapping = {}
        used = set()
        for local_index, part in enumerate(structure['parts']):
            if not any(row[1] == local_index for row in structure['rows']):
                continue
            name = re.sub(r'[^\w]', '', part['name'].casefold())
            # A repeated full system keeps its track identities at page turns,
            # even if a continuation page misreads or omits instrument names.
            match = identities[template][order.index(local_index)] if template in identities else None
            if match is None:
                match = next((i for i in by_name.get(name, []) if i not in used), None)
            if match is None and local_index < len(parts) and local_index not in used:
                old = parts[local_index]
                if old['instrument'] == part['instrument'] and old.get('strings') == part.get('strings') and old['program'] == part['program']:
                    match = local_index
            if match is None:
                match = len(parts)
                parts.append({**part, 'id': f'part-{match + 1}'})
            by_name.setdefault(name, [])
            if match not in by_name[name]:
                by_name[name].append(match)
            used.add(match)
            mapping[local_index] = match
        if template is not None:
            identities[template] = [mapping[i] for i in order]
        systems = defaultdict(list)
        for boxes, (system, part, staff) in zip(rows, structure['rows'], strict=True):
            systems[system].append((boxes, mapping[part], staff))
        for system, members in sorted(systems.items()):
            reference = max((boxes for boxes, _, _ in members), key=len)
            for boxes, part, staff in members:
                profile = parts[part]
                for row, column in zip(boxes, aligned_columns(boxes, reference), strict=True):
                    row.update(part_id=profile['id'], staff_id=f'staff-{staff + 1}',
                               part_name=profile['name'], instrument=profile['instrument'],
                               midi_program=int(profile['program']), string_count=profile.get('strings'),
                               row_index=row.get('row_index', row['system_index']),
                               system_index=system, bar_index=bar_offset + column)
            bar_offset += len(reference)
        predictions.append({'page': page_number, 'image': path, 'raw': raw,
                            'systems_from_image': systems_from_image, 'parsed': structure})
    return parts, predictions
