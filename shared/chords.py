"""Visible chord symbols and diagrams; string arrays run low to high."""

import re
import unicodedata
from urllib.parse import quote, unquote


def normalize_diagram(value):
    if not isinstance(value, dict):
        return None
    frets = value.get('frets')
    if not isinstance(frets, list) or not 1 <= len(frets) <= 12:
        return None
    if any(f != 'x' and (type(f) is not int or not 0 <= f <= 36) for f in frets):
        return None
    base = value.get('base_fret', 1)
    if type(base) is not int or not 1 <= base <= 36:
        return None
    if any(type(f) is int and 0 < f < base for f in frets):
        return None
    fingers = value.get('fingers')
    if fingers is None:
        fingers = [None] * len(frets)
    if (not isinstance(fingers, list) or len(fingers) != len(frets)
            or any(f is not None and (type(f) is not int or not 0 <= f <= 4) for f in fingers)):
        return None
    barres = value.get('barres', [])
    if not isinstance(barres, list):
        return None
    for barre in barres:
        if not isinstance(barre, list) or len(barre) != 3 or any(type(v) is not int for v in barre):
            return None
        fret, low, high = barre
        if not base <= fret <= 36 or not 0 <= low < high < len(frets):
            return None
    return {'base_fret': base, 'frets': list(frets), 'fingers': list(fingers), 'barres': barres}


def diagram_effect(diagram):
    diagram = normalize_diagram(diagram)
    if diagram is None:
        raise ValueError('Invalid chord diagram')
    return 'diagram:' + ':'.join([
        str(diagram['base_fret']), '/'.join(map(str, diagram['frets'])),
        '/'.join('-' if f is None else str(f) for f in diagram['fingers']),
        ';'.join('/'.join(map(str, b)) for b in diagram['barres']) or '-',
    ])


def parse_diagram_effect(effect):
    try:
        _, base, frets, fingers, barres = effect.split(':')
        return normalize_diagram({'base_fret': int(base),
            'frets': ['x' if f == 'x' else int(f) for f in frets.split('/')],
            'fingers': [None if f == '-' else int(f) for f in fingers.split('/')],
            'barres': [] if barres == '-' else [[int(v) for v in b.split('/')] for b in barres.split(';')]})
    except (ValueError, TypeError):
        return None


def chord_key(name):
    return re.sub(r'\s+', '', str(name or '')).replace('♭', 'b').replace('♯', '#')


def chord_recognition_errors(measure):
    """Request a visual retry for prose or malformed harmonic symbols.

    This is an OCR confidence check, not an IR restriction: custom names remain
    editable, and an unresolved annotation must not discard musical events.
    """
    accidental = r'(?:##|bb|#|b|x)?'
    root = rf'(?:[A-Ha-h]{accidental}|{accidental}(?:[IViv]+|[1-7]))'
    quality = r'(?:[mM]aj|[mM]in|dim|aug|sus|dom|m|M|\+|-|°|ø|Δ|o)?'
    extension = r'(?:2|4|5|6|7|9|11|13)?'
    modifier = r'(?:[#b](?:5|6|7|9|11|13)|(?:[mM]aj|M)(?:7|9|11|13)|(?:sus|add|omit|no)(?:2|3|4|5|6|7|9|11|13)|alt)'
    suffix = rf'{quality}{extension}(?:M|\+|-|°|ø)?(?:{modifier}){{0,4}}'
    group = rf'(?:\((?:{modifier}|{extension})(?:,?(?:{modifier}|{extension})){{0,3}}\))?'
    pattern = rf'{root}{suffix}{group}(?:/(?:{root}|9|11|13))?'
    errors = []
    for voice in measure.get('voices', []):
        for event in voice.get('events', []):
            name, _ = event_chord(event)
            if name is None:
                continue
            normalized = chord_key(unicodedata.normalize('NFKC', name)).replace('major', 'maj').replace('minor', 'min')
            if normalized.lower() in {'nc', 'n.c', 'n.c.', 'nochord'}:
                continue
            repeated = re.search(r'((?:[#b]|sus|add|omit|no)(?:2|3|4|5|6|7|9|11|13))\1', normalized)
            if not re.fullmatch(pattern, normalized) or repeated:
                errors.append(
                    'Uncertain chord symbol ' + repr(name[:64]) + ': re-read its visible name and onset; '
                    'titles, section labels, instrument names and technique instructions belong in text, '
                    'not chord. Keep all musical events and do not invent a replacement chord')
    return errors


def event_chord(event):
    effects = event.get('effects', [])
    name = next((unquote(e[6:]) for e in effects if e.startswith('chord:')), None)
    diagram = next((parse_diagram_effect(e) for e in effects if e.startswith('diagram:')), None)
    return name, diagram


def merge_chord_review(original, reread, mode):
    """Apply annotation-only rereading without regenerating valid music."""
    from shared.m2 import parse_measure_target, format_measure_target

    measure, candidate = parse_measure_target(original), parse_measure_target(reread)
    events = {(voice['voice'], event['start']): event
              for voice in candidate['voices'] for event in voice['events']}
    positions = {(voice['voice'], event['start'])
                 for voice in measure['voices'] for event in voice['events']}
    if any(event_chord(event)[0] and at not in positions for at, event in events.items()):
        raise ValueError('The reread chord onset does not match an existing musical event')
    for voice in measure['voices']:
        for event in voice['events']:
            replacement = events.get((voice['voice'], event['start']))
            if replacement is not None:
                event['effects'] = [e for e in event.get('effects', [])
                                    if not e.startswith(('chord:', 'diagram:', 'text:'))]
                event['effects'].extend(e for e in replacement.get('effects', [])
                                       if e.startswith(('chord:', 'diagram:', 'text:')))
    if candidate.get('section'):
        measure['section'] = candidate['section']
    return format_measure_target(measure, mode, preserve_playback=True)


def retain_uncertain_chords_as_text(target, mode):
    """Keep unresolved OCR text visible without declaring a musical chord.

    Called only after visual retries fail. Manual/custom chord editing is not
    restricted by this confidence policy.
    """
    from shared.m2 import parse_measure_target, format_measure_target

    measure, uncertain = parse_measure_target(target), []
    for voice in measure['voices']:
        for event in voice['events']:
            if not chord_recognition_errors({'voices': [{'events': [event]}]}):
                continue
            name, _ = event_chord(event)
            uncertain.append({'voice': voice['voice'], 'start': event['start'], 'text': name})
            captions = [unquote(e[5:]) for e in event.get('effects', []) if e.startswith('text:')]
            event['effects'] = [e for e in event.get('effects', [])
                                if not e.startswith(('chord:', 'text:'))]
            event['effects'].append('text:' + quote('; '.join(dict.fromkeys([*captions, name])), safe=''))
    return (format_measure_target(measure, mode, preserve_playback=True) if uncertain else target), uncertain


def diagram_for_strings(diagram, string_ids):
    """Project a diagram onto a GP5 track's zero-based high-to-low strings."""
    count = len(diagram['frets'])
    indexes = [count - 1 - i for i in reversed(string_ids)]
    barres = []
    for fret, low, high in diagram['barres']:
        covered = [j for j, i in enumerate(indexes) if low <= i <= high]
        if len(covered) > 1:
            barres.append([fret, covered[0], covered[-1]])
    return {'base_fret': diagram['base_fret'],
            'frets': [diagram['frets'][i] if 0 <= i < count else 'x' for i in indexes],
            'fingers': [diagram['fingers'][i] if 0 <= i < count else None for i in indexes],
            'barres': barres}


def attach_chord_annotations(records, predictions):
    """Retain page evidence and enrich timed symbols when their names agree.

    A page-header diagram is a library entry, not a new chord at beat zero.
    Beat timing comes from measure recognition, never horizontal interpolation.
    """
    from shared.m2 import parse_measure_target, format_measure_target

    annotations = [p for p in predictions if p.get('parsed', {}).get('kind') in {'chord', 'chord_diagram'}]
    for row in records:
        measure = parse_measure_target(row['target'])
        changed = False
        part_annotations = [p for p in annotations
                            if p.get('part_id', 'part-1') == row.get('part_id', 'part-1')]
        page_annotations = [p for p in part_annotations if p.get('page') == row.get('page')]
        x, y, w, h = row.get('bbox', [0, 0, 0, 0])
        local = []
        for p in page_annotations:
            a, b, c, d = p.get('bbox', [0, 0, 0, 0])
            if x - 8 <= a + c / 2 <= x + w + 8 and y - max(50, h * .65) <= b + d <= y + h:
                local.append(p)
        if local:
            row['chord_annotations'] = local
        for voice in measure['voices']:
            for event in voice['events']:
                name, diagram = event_chord(event)
                if not name or diagram:
                    continue
                def matching(values):
                    return [p for p in values
                            if chord_key(p['parsed'].get('text')) == chord_key(name)
                            and (shape := normalize_diagram(p['parsed'].get('diagram')))
                            and (not row.get('tuning') or len(shape['frets']) == len(row['tuning']))]
                candidates = matching(local) or matching(part_annotations)
                diagrams = {diagram_effect(p['parsed']['diagram']) for p in candidates}
                if len(diagrams) == 1:
                    event['effects'].append(diagrams.pop())
                    changed = True
        if changed:
            row['target'] = format_measure_target(measure, row['mode'], preserve_playback=True)
    return records
