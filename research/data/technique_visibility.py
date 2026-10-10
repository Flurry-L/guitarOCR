"""Source-evidenced visual supervision; never alter source playback labels."""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from scorelib.score import format_measure_target, parse_measure_target

from research.common.pdf import open_pdf
from research.data.catalog import native_document


def has_technique(target, effect):
    """Inspect actual effect fields, never a free-text/chord-name substring."""
    # Fast rejection keeps corpus scans cheap. A positive result always requires
    # parsing the score: e.g. text:pm or chord:pm cannot become a PM technique.
    if effect not in target:
        return False
    return any(effect in owner.get('effects', ())
               for voice in parse_measure_target(target)['voices']
               for event in voice['events'] for owner in (event, *event.get('notes', ())))


@lru_cache(maxsize=8192)
def native_pm_evidence(label_path, mode):
    """Verify a native notation export against its own source and TAB PDF.

    A missing word alone is insufficient: require the declared native profile,
    extractable notation text and a matching TAB export that prints PM. Other
    renderers do not acquire an absence label from a renderer/mode name.
    """
    label = Path(label_path)
    if mode != 'notation':
        return None
    unknown = {'pm': {'state': 'unknown', 'rule': 'native_pdf_evidence',
                       'source_label': str(label)}}
    if not label.is_file():
        return unknown
    source = json.loads(label.read_text())
    if not source.get('statistics', {}).get('technique_counts', {}).get('pm'):
        return unknown
    exports = {}
    for display in ('notation', 'tab'):
        layouts = list((native_document(label, display) / 'tracks').glob('*/layout.json'))
        if len(layouts) != 1:
            return unknown
        layout_path = layouts[0]
        profile = json.loads(layout_path.read_text())
        pdf_path = layout_path.with_name('score.pdf')
        if (profile.get('schema') != 'gpomr.render-layout' or not profile.get('pdf_ok')
                or profile.get('display_mode') != display or not pdf_path.is_file()):
            return unknown
        with open_pdf(pdf_path) as document:
            words = [word[4] for page in document for word in page.words()]
        tokens = [''.join(c for c in word.casefold() if c.isalnum()) for word in words]
        if not any(tokens):
            return unknown
        count = sum(token in {'pm', 'palmmute'} for token in tokens)
        count += sum(pair == ('palm', 'mute') for pair in zip(tokens, tokens[1:]))
        exports[display] = {'pm_text_count': count}
    # A printed word somewhere in a score cannot establish its event coverage.
    # Keep such playback targets for review unless a row has explicit scope.
    state = ('absent' if not exports['notation']['pm_text_count']
             and exports['tab']['pm_text_count'] else 'unknown')
    return {'pm': {'state': state, 'rule': 'native_pdf_evidence',
                   'reason': ('notation_text_present_without_event_scope'
                              if exports['notation']['pm_text_count']
                              else 'notation_absent_with_tab_control' if state == 'absent'
                              else 'missing_positive_tab_control'),
                   'source_label': str(label.resolve()),
                   'renders': exports}}


def visible_target(target, mode, renderer, *, printed_techniques=(), visibility_evidence=None):
    """Remove a playback PM flag only with explicit source-visibility evidence."""
    if renderer != 'guitar_pro' or mode != 'notation' or 'pm' in printed_techniques:
        return target, {}
    if not visibility_evidence or visibility_evidence.get('pm', {}).get('state') != 'absent':
        return target, {}
    if not has_technique(target, 'pm'):
        return target, {}
    measure = parse_measure_target(target)
    removed = Counter()
    for voice in measure['voices']:
        for event in voice['events']:
            for owner in [event, *event.get('notes', [])]:
                effects = owner.get('effects', [])
                removed['pm'] += effects.count('pm')
                owner['effects'] = [effect for effect in effects if effect != 'pm']
    if not removed['pm']:
        return target, {}
    return format_measure_target(measure, mode, preserve_playback=True), dict(removed)


def prepare_visual_techniques(row):
    """Return a record with visual target and inspectable, idempotent provenance."""
    result = dict(row)
    if (row['mode'] == 'notation' and has_technique(row['target'], 'pm')
            and row.get('renderer') in (None, 'guitar_pro')
            and 'pm' not in row.get('printed_techniques', ())):
        if 'technique_evidence' not in result:
            result['technique_evidence'] = native_pm_evidence(
                str(row.get('label_json') or ''), row['mode'])
        if result['technique_evidence']['pm']['state'] == 'absent':
            result['renderer'] = 'guitar_pro'
    target, removed = visible_target(row['target'], row['mode'], result.get('renderer'),
                                     printed_techniques=row.get('printed_techniques', ()),
                                     visibility_evidence=result.get('technique_evidence'))
    if removed:
        result.setdefault('source_target', row['target'])
        result['target'] = target
        result['technique_visibility'] = {'renderer': result['renderer'], 'mode': row['mode'],
                                           'removed_hidden_effects': removed,
                                           'rule': result['technique_evidence']['pm'].get(
                                               'rule', 'explicit_visibility_evidence')}
        if 'target_utf8_bytes' in result:
            result['target_utf8_bytes'] = len(target.encode('utf-8'))
    return result
