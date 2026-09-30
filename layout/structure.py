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
PAIRING_HINT = (' If notation and TAB of one instrument have separate blue row labels, '
                'assign both rows the SAME system, part and staff indices. '
                'Different printed part names must keep different part indices, '
                'including Guitar and Guitar II or multiple parts of the same instrument. '
                'A row that already contains notation+TAB cannot pair with another TAB row.')


def structure_prompt(count, modes=None, systems=None):
    text = STRUCTURE_PROMPT + PAIRING_HINT + f'There are {count} marked rows.'
    if modes and 'notation' in modes and 'tab' in modes:
        text += ' Detected row types: ' + ', '.join(f'R{i + 1}={mode}' for i, mode in enumerate(modes)) + '.'
        if systems is not None:
            text += f' Continuous barlines connect the rows into systems {systems}.'
    return text

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


def structure_schema(row_count, systems=None):
    schema = deepcopy(STRUCTURE_SCHEMA)
    schema['properties']['rows'].update(minItems=row_count, maxItems=row_count)
    schema['properties']['parts']['maxItems'] = row_count
    if systems is not None and len(systems) == row_count:
        # Exact lengths already forbid trailing items. Omitting `items`
        # also keeps the tuple compatible with llama.cpp's schema compiler.
        schema['properties']['rows'].pop('items')
        schema['properties']['rows'].update(prefixItems=[{
            'type': 'array', 'minItems': 3, 'maxItems': 3,
            'prefixItems': [{'const': system},
                            {'type': 'integer', 'minimum': 0, 'maximum': row_count - 1},
                            {'type': 'integer', 'minimum': 0, 'maximum': 15}],
        } for system in systems])
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
        # Antialiasing can leave only one thin staff line dark enough.
        # It locates the staff; the continuous bracket still has to pass the
        # separate 90% vertical-coverage check below.
        # Very thin notation lines may disappear when the marked page is
        # resized. The row label still anchors its detector box; the system
        # bracket must independently cover the interval below.
        staff.append(float(np.median(hits)) if len(hits) else center)
    systems = [0]
    for first, second in zip(staff, staff[1:]):
        region = ink[round(first):round(second), :int(rgb.shape[1] * .23)]
        if not region.size:
            return None
        score = float(region.mean(0).max())
        if .7 < score < .9:
            # Two disconnected brackets may cover much of a tall interval.
            # A substantial continuous blank separates systems; scattered
            # missing pixels could instead be a damaged connecting line.
            missing = np.flatnonzero(~region[:, int(region.mean(0).argmax())])
            gaps = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
            if max((len(gap) for gap in gaps), default=0) < max(12, len(region) * .08):
                return None
        systems.append(systems[-1] + int(score < .9))
    return systems


def compatible_templates(parts, modes):
    """Find contiguous part layouts compatible with the detected staff types."""
    candidates = []

    def visit(part_index, position, template):
        if len(candidates) > 1:
            return
        if part_index == len(parts):
            if position == len(modes):
                candidates.append(template)
            return
        part = parts[part_index]
        if part['instrument'] in {'guitar', 'bass'}:
            patterns = [(('both',), (0,)), (('tab',), (0,)),
                        (('notation',), (0,)), (('notation', 'tab'), (0, 0)),
                        (('tab', 'notation'), (0, 0))]
        else:
            patterns = [(('notation',), (0,))]
            if (part['instrument'] == 'pitched' and
                    (re.search(r'\b(?:piano|keyboard|organ|harp)\b', part['name'], re.I)
                     or part['program'] in {*range(8), *range(16, 21), 46})):
                patterns.append((('notation', 'notation'), (0, 1)))
        for pattern, staves in patterns:
            end = position + len(pattern)
            if tuple(modes[position:end]) == pattern:
                visit(part_index + 1, end, template + [(part_index, staff) for staff in staves])

    visit(0, 0, [])
    return candidates


def constrain_rows(value, systems, modes=None):
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
    grand = {i for i, part in enumerate(parts)
             if re.search(r'\b(?:piano|keyboard|organ|harp)\b', part['name'], re.I)}
    valid_template = (len(template) == width
                      and {p for p, _s in template} == set(range(len(parts)))
                      and all(0 <= staff < 16 for _p, staff in template))
    if valid_template:
        occurrences = defaultdict(list)
        first_indices = next(iter(groups.values()))
        for index, slot in enumerate(template):
            occurrences[slot].append(index)
        valid_template = all(len(indices) == 1 or (
            modes is not None and len(indices) == 2
            and {modes[first_indices[i]] for i in indices} == {'notation', 'tab'}
        ) for indices in occurrences.values())
    if not valid_template and modes is not None:
        patterns = {tuple(modes[i] for i in indices) for indices in groups.values()}
        if len(patterns) == 1:
            candidates = compatible_templates(parts, patterns.pop())
            if len(candidates) == 1:
                value['parts'] = parts
                value['rows'] = [[system, part, staff] for system in groups
                                 for part, staff in candidates[0]]
                return
    if valid_template:
        value['parts'] = parts
        value['rows'] = [[system, part, staff] for system in groups for part, staff in template]
        return
    if not valid_template and sum(2 if i in grand else 1 for i in range(len(parts))) != width:
        # Printed names may contain OCR spelling errors. The predicted MIDI
        # family can complete a grand-staff template when its total row count
        # agrees with every geometrically detected system.
        grand |= {i for i, part in enumerate(parts)
                  if part['instrument'] == 'pitched' and part['program'] in {*range(8), *range(16, 21), 46}}
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


def separate_fretted_lanes(value, modes):
    """Retain simultaneous fretted rows that cannot be one notation/TAB pair."""
    rows, parts = value['rows'], value['parts']
    if modes is None or len(rows) != len(modes) or not all(
            isinstance(r, list) and len(r) == 3 and all(type(v) is int for v in r)
            and 0 <= r[1] < len(parts) for r in rows):
        return
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[0]].append(index)
    templates = {tuple((rows[i][1], rows[i][2], modes[i]) for i in indices)
                 for indices in groups.values()}
    if len(templates) != 1:
        return
    first = next(iter(groups.values()))
    slots = defaultdict(list)
    for i in first:
        slots[tuple(rows[i][1:])].append(i)
    duplicated = {slot for slot, indices in slots.items() if len(indices) > 1
                  and parts[slot[0]]['instrument'] in {'guitar', 'bass'}
                  and not (len(indices) == 2 and {modes[i] for i in indices} == {'notation', 'tab'})}
    if not duplicated:
        return
    # Repeated notation/TAB layers need a fresh visual reading; only separate
    # rows whose individually detected mode already represents a whole staff.
    if any({'notation', 'tab'}.issubset({modes[i] for i in slots[slot]}) for slot in duplicated):
        return
    profiles, mapping, occurrence, template = [], {}, Counter(), []
    for i in first:
        _, part, staff = rows[i]
        slot = (part, staff)
        lane = occurrence[slot] if slot in duplicated else 0
        occurrence[slot] += 1
        key = (part, lane)
        if key not in mapping:
            mapping[key] = len(profiles)
            profile = deepcopy(parts[part])
            if lane:
                profile['name'] += f' ({lane + 1})'
            profiles.append(profile)
        template.append((mapping[key], staff))
    value.update(parts=profiles, rows=[[system, part, staff] for system in groups for part, staff in template],
                 separated_fretted_lanes=True)


def parse_structure(raw, count, systems=None, modes=None):
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
        constrain_rows(value, systems, modes)
        parts, rows = value['parts'], value['rows']
    elif modes is not None and len(rows) == count and all(
            isinstance(row, list) and len(row) == 3 and all(type(v) is int for v in row) for row in rows):
        # The predicted system boundaries can still support an unambiguous
        # repair of invalid part/staff assignments. Keep those boundaries;
        # do not infer new systems from a guessed number of instruments.
        groups = defaultdict(list)
        seen = defaultdict(list)
        invalid = False
        for i, row in enumerate(rows):
            groups[row[0]].append(i)
            seen[tuple(row)].append(modes[i])
            invalid |= not 0 <= row[1] < len(parts)
        invalid |= any(len(values) > 1 and (len(values) != 2 or set(values) != {'notation', 'tab'})
                       for values in seen.values())
        patterns = {tuple(modes[i] for i in indices) for indices in groups.values()}
        if invalid and len(patterns) == 1:
            candidates = compatible_templates(parts, patterns.pop())
            if len(candidates) == 1:
                rows = value['rows'] = [[system, part, staff] for system in groups
                                       for part, staff in candidates[0]]
    separate_fretted_lanes(value, modes)
    if value.get('separated_fretted_lanes'):
        # Separating two guitars can also make the remaining piano grand
        # staff template unique; resolve its two hands with the same rules.
        constrain_rows(value, systems if systems is not None else [r[0] for r in value['rows']], modes)
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
    seen = {}
    for index, (system, part, staff) in enumerate(rows):
        if any(type(x) is not int for x in (system, part, staff)):
            raise ValueError('Structure indices must be integers')
        if not 0 <= part < len(parts):
            raise ValueError(f'R{index + 1} uses nonexistent part {part}; parts must be 0..{len(parts) - 1}. '
                             'Separate notation and TAB rows of one instrument share the same part and staff')
        if system < 0 or system < previous or not 0 <= staff < 16:
            raise ValueError(f'R{index + 1} has invalid system/staff indices; systems must stay in page order')
        if (system, part, staff) in seen:
            previous_index = seen[(system, part, staff)]
            if (modes is None or previous_index is None
                    or {modes[index], modes[previous_index]} != {'notation', 'tab'}):
                raise ValueError(f'R{index + 1} repeats system {system}, part {part}, staff {staff}; '
                                 'only complementary notation/TAB rows may share this triple')
            seen[(system, part, staff)] = None
        else:
            seen[(system, part, staff)] = index
        previous = system
    if modes is not None:
        for part_index, part in enumerate(parts):
            visible = [mode for mode, row in zip(modes, rows, strict=True) if row[1] == part_index]
            if visible and all(mode == 'notation' for mode in visible):
                part['strings'] = None
    return value


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
    """Recover incomplete page assignments from another visible complete system."""
    if systems is None:
        return None
    invalid_rows = False
    try:
        incomplete = parse_structure(raw, len(rows), modes=[Counter(r['mode'] for r in boxes).most_common(1)[0][0] for boxes in rows])
    except (ValueError, KeyError, TypeError, IndexError):
        # A missing part often makes the model assign two unrelated rows to
        # one staff. Its readable part names can still identify a unique
        # compatible template on another page; the invalid assignments must
        # not prevent that recovery.
        try:
            value = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
            used = sorted({r[1] for r in value['rows']
                           if isinstance(r, list) and len(r) == 3 and type(r[1]) is int
                           and 0 <= r[1] < len(value['parts'])})
            # Invalid row indices are discarded only for extracting readable
            # profiles. Recovery still requires one complete reference page
            # with the same visible staff modes and all observed identities.
            if not used:
                return None
            profiles = [value['parts'][i] for i in used]
            incomplete = parse_structure(json.dumps({
                'parts': profiles, 'rows': [[i, i, 0] for i in range(len(profiles))],
            }), len(profiles))
            invalid_rows = True
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
        if ((len(profiles) <= len(incomplete['parts']) and not invalid_rows)
                or not observed.issubset({identity(p) for p in profiles})):
            continue
        key = (tuple(identity(p) for p in profiles), template)
        candidates[key] = {'parts': deepcopy(profiles), 'rows': [
            [system, part, staff] for system in groups for part, staff, _mode in template], 'reference_page': page}
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return None


def read_row_profiles(rows, systems, output, backend):
    """Enlarge visible row labels when full-page structure retries fail."""
    from document_info.prompts import STAFF_PROMPT, STAFF_SCHEMA
    from document_info.image_ocr import parse_info_response
    from document_info.staff_image import focus_staff

    indices = [i for i in range(len(rows)) if systems is None or systems[i] == systems[0]][:16]
    requests, paths = [], []
    for index in indices:
        boxes = rows[index]
        top = min(r['bbox'][1] for r in boxes)
        bottom = max(r['bbox'][1] + r['bbox'][3] for r in boxes)
        path = Path(output) / f'row-{index + 1}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(boxes[0]['source_page']) as page:
            focus_staff(page.crop((0, max(0, int(top) - 12), page.width,
                                  min(page.height, int(bottom) + 13)))).save(path)
        paths.append(path)
        requests.append([{'role': 'user', 'content': [
            {'type': 'image', 'url': str(path.resolve())}, {'type': 'text', 'text': STAFF_PROMPT}]}])
    options = {'json_schema': STAFF_SCHEMA} if getattr(backend, 'supports_json_schema', False) else {}
    results = (backend.generate_batch(requests, 256, **options) if hasattr(backend, 'generate_batch') else
               [backend.generate(message, 256, **options) for message in requests])
    return [{'row': f'R{index + 1}', **parse_info_response(raw, 'staff', path)}
            for index, path, (raw, _tokens) in zip(indices, paths, results, strict=True)]


def read_structure(source, output, backend, cancelled=None, *, compact=False):
    from shared.tasks import Cancelled

    structured = getattr(backend, 'supports_json_schema', False)
    grouped = defaultdict(list)
    for row in staff_rows(source['records']):
        grouped[row[0]['page']].append(row)
    requests = []
    geometry, row_modes = {}, {}
    for page_number, rows in sorted(grouped.items()):
        path = structure_image(rows[0][0]['source_page'], rows, Path(output) / f'structure-{page_number}.png')
        geometry[page_number] = marked_systems(path, len(rows))
        row_modes[page_number] = [Counter(r['mode'] for r in row).most_common(1)[0][0] for row in rows]
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'url': structure_model_image(path) if compact else path},
            {'type': 'text', 'text': structure_prompt(len(rows), row_modes[page_number], geometry[page_number])}]}]
        requests.append((page_number, path, messages))
    initial = {}
    for offset in range(0, len(requests), 4):
        if cancelled and cancelled():
            raise Cancelled('已取消总谱结构识别')
        batch = requests[offset:offset + 4]
        messages = [r[2] for r in batch]
        generation = {'json_schema': [structure_schema(len(grouped[r[0]]), geometry[r[0]]) for r in batch]} if structured else {}
        outputs = backend.generate_batch(messages, 2048, **generation) if hasattr(backend, 'generate_batch') else [backend.generate(m, 2048, **generation) for m in messages]
        for request, (raw, _tokens) in zip(batch, outputs, strict=True):
            initial[request[0]] = (*request[1:], raw)
    parsed, resolved_inputs = {}, {}
    for page_number, rows in sorted(grouped.items()):
        if cancelled and cancelled():
            raise Cancelled('已取消总谱结构识别')
        path, messages, raw = initial[page_number]
        systems_from_image = geometry[page_number]
        error = None
        attempts = []
        generation = {'json_schema': structure_schema(len(rows), systems_from_image)} if structured else {}
        for _attempt in range(3):
            if _attempt == 2:
                # Re-read labels at staff scale and restore full-page context.
                # The same OCR model supplies visual evidence for omitted
                # parts; geometry still determines their system and staff.
                hint = (f' Visible barlines give system indices {systems_from_image} for these rows.'
                        if systems_from_image is not None else '')
                profiles = read_row_profiles(rows, systems_from_image,
                                             Path(output) / f'structure-{page_number}-labels', backend)
                hint += (' Enlarged row-label readings: ' + json.dumps(profiles, ensure_ascii=False)
                         + '. Recheck these identities against the page. Include every visible part; '
                         'a left-hand staff belongs to its piano, not to the preceding instrument.')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'url': path},
                    {'type': 'text', 'text': structure_prompt(len(rows), row_modes[page_number], systems_from_image) + hint},
                ]}]
            if _attempt:
                raw, _ = backend.generate(messages, 2048, **generation)
            try:
                structure = parse_structure(raw, len(rows), systems_from_image, row_modes[page_number])
                break
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                attempts.append({'raw': raw, 'error': error})
                references = []
                for other, reference_rows in grouped.items():
                    if other == page_number:
                        continue
                    try:
                        reference = parsed.get(other) or parse_structure(initial[other][2], len(reference_rows), geometry[other], row_modes[other])
                    except (ValueError, KeyError, TypeError, IndexError):
                        continue
                    references.append((other, reference, reference_rows))
                structure = complete_from_reference(raw, rows, systems_from_image, references)
                if structure is not None:
                    break
                messages.extend([{'role': 'assistant', 'content': [{'type': 'text', 'text': raw}]},
                                 {'role': 'user', 'content': [{'type': 'text', 'text': error + '. Return the complete corrected JSON.'}]}])
        else:
            (Path(output) / f'structure-{page_number}-errors.json').write_text(
                json.dumps(attempts, ensure_ascii=False, indent=2))
            raise ValueError(f'Cannot resolve score structure on page {page_number}: {error}')
        parsed[page_number] = structure
        resolved_inputs[page_number] = (path, raw, systems_from_image, _attempt + 1)
    reconcile_page_profiles([(parsed[page], rows) for page, rows in sorted(grouped.items())])
    predictions, parts, by_name, identities = [], [], {}, {}
    resolved_records = []
    bar_offset = 0
    for page_number, rows in sorted(grouped.items()):
        structure = parsed[page_number]
        path, raw, systems_from_image, attempts = resolved_inputs[page_number]
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
            from layout.score_grid import align_system, fuse_paired_staves
            aligned, column_count = align_system(members, output, page_number, system)
            aligned = fuse_paired_staves(aligned, output, page_number, system)
            for boxes, part, staff in aligned:
                profile = parts[part]
                for row, column in boxes:
                    row.update(part_id=profile['id'], staff_id=f'staff-{staff + 1}',
                               part_name=profile['name'], instrument=profile['instrument'],
                               midi_program=int(profile['program']), string_count=profile.get('strings'),
                               row_index=row.get('row_index', row['system_index']),
                               system_index=system, bar_index=bar_offset + column)
                    resolved_records.append(row)
            bar_offset += column_count
        predictions.append({'page': page_number, 'image': path, 'raw': raw,
                            'systems_from_image': systems_from_image, 'parsed': structure,
                            'attempts': attempts})
    for number, row in enumerate(resolved_records, 1):
        row['measure_number'] = number
    source['records'] = resolved_records
    return parts, predictions
