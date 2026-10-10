"""The visible-only contract at real musical boundaries, without model/GPU use."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from research.common.pdf import open_pdf
from research.data.native_supervision import (
    native_connection_evidence, native_scope_evidence, native_single_slides,
    native_observations, native_tuplet_evidence,
)
from research.inference.assembly import assemble, parse_observation


def measure(index, frets=(3, 3, 3, 3), relations=(), *, system=0, voices=None):
    if voices is None:
        voices = [{'voice': 0, 'events': [
            {'start': i * 960, 'duration': 'q', 'status': 'normal',
             'notes': [{'string': 1, 'fret': fret}]} for i, fret in enumerate(frets)]}]
    return {'source_id': 'score', 'part_id': 0, 'staff_id': 0, 'bar_index': index,
            'mode': 'tab', 'page': 1, 'system_index': system,
            'observation': {'state': {'time': {'status': 'observed', 'value': '4/4'}},
                            'voices': voices, 'relations': list(relations)}}


def connection(kind, start, end, **extra):
    return dict(kind=kind, start=start, end=end, **extra)


def note(result, record, event, voice=0):
    voices = result['records'][record]['resolved']['voices']
    return next(v for v in voices if v['voice'] == voice)['events'][event]['notes'][0]


class EvidencePage:
    def __init__(self, *, curves=(), chars=(), dashes=(), words=(), lines=()):
        self.drawing = SimpleNamespace(curves=list(curves), chars=list(chars), lines=list(lines))
        self.dashes = list(dashes)
        self.printed_words = list(words)

    def _text_page(self):
        return self.drawing

    def dashed_lines(self):
        return self.dashes

    def words(self):
        return self.printed_words


def glyph(x0, x1, *, y=40, string=0):
    scale = 25.4 / 72
    return {'bbox_mm': [x0 * scale, y * scale, (x1 - x0) * scale, 4 * scale],
            'native_string_index': string,
            'render_origin_mm': [(x0 + x1) / 2 * scale, (y + 2) * scale]}


class ObservationAssemblyTests(unittest.TestCase):
    def test_signature_classifier_absence_prevents_false_state_changes(self):
        rows = [measure(0), measure(1), measure(2)]
        rows[0]['observation']['state'].update(time={'status': 'observed', 'value': '6/8'},
            key={'status': 'absent'}, clef={'status': 'observed', 'value': 'G2'})
        rows[1]['observation']['state'].update(time={'status': 'observed', 'value': '11/1'},
            key={'status': 'observed', 'value': 1})
        rows[2]['observation']['state'] = {}
        absent = {'status': 'absent', 'source': 'visual_classifier', 'calibrated': True}
        extra = [{'staff_start': True}, {'state': {'time': absent, 'key': absent}}, {}]
        result = assemble(rows, auxiliary=extra)
        for row in result['records']:
            self.assertEqual(row['resolved']['state']['time']['value'], '6/8')
            self.assertEqual(row['resolved']['state']['key']['value'], 0)
        self.assertEqual(result['records'][1]['observation']['state']['time']['value'], '11/1')
        rows[0]['bar_index'] = 7
        self.assertIsNone(assemble(rows[:1], auxiliary=extra[:1])['records'][0]['resolved']['state']['key']['value'])
        rows[0]['bar_index'] = 0
        rows[0]['observation']['state']['key'] = {'status': 'unknown'}
        self.assertIsNone(assemble(rows[:1], auxiliary=extra[:1])['records'][0]['resolved']['state']['key']['value'])

    def test_omitted_state_continues_but_unreadable_state_interrupts(self):
        rows = [measure(i) for i in range(4)]
        rows[0]['observation']['state']['clef_octave'] = {'status': 'observed', 'value': 0}
        rows[2]['observation']['state']['clef_octave'] = {'status': 'unknown'}
        result = assemble(rows)
        self.assertEqual(result['records'][1]['resolved']['state']['clef_octave'],
                         {'status': 'inherited', 'value': 0})
        self.assertIsNone(result['records'][2]['resolved']['state']['clef_octave']['value'])
        self.assertIsNone(result['records'][3]['resolved']['state']['clef_octave']['value'])
        rows[2]['observation']['state']['clef_octave'] = {'status': 'observed', 'value': 7}
        result = assemble(rows)
        self.assertIsNone(result['records'][2]['resolved']['state']['clef_octave']['value'])

    def test_dual_staves_match_independent_notes_after_score_context(self):
        row = measure(0, voices=[{'voice': 0, 'events': [
            {'start': 0, 'duration': 'w', 'status': 'normal',
             'notes': [{'string': 1, 'fret': 0}, {'string': 2, 'fret': 10}],
             'notation_notes': [{'staff_position': 10, 'accidental': None},
                                {'staff_position': 7, 'accidental': None}]}]}])
        row['mode'] = 'both'
        values = dict(time='4/4', key=0, clef='G2', clef_octave=0,
            tuning=[64, 59, 55, 50, 45, 40], instrument_transpose=-12,
            instrument_transpose_explicit=False, capo=0)
        extra = {'state': {k: {'status': 'observed', 'value': v} for k, v in values.items()}}
        result = assemble([row], auxiliary=[extra])
        self.assertIn('s1f0p76', result['records'][0]['ir'])
        self.assertIn('s2f10p81', result['records'][0]['ir'])
        self.assertEqual([n['staff_position'] for n in result['records'][0]['resolved']['voices'][0]['events'][0]['notes']], [7, 10])
        self.assertEqual(result['records'][0]['observation'], row['observation'])
        event = row['observation']['voices'][0]['events'][0]
        event['notes'][1]['fret'] = None
        result = assemble([row], auxiliary=[extra])
        self.assertIn('s2f10p81', result['records'][0]['ir'])
        self.assertIsNone(result['records'][0]['observation']['voices'][0]['events'][0]['notes'][1]['fret'])
        event['notes'][0]['fret'] = None
        result = assemble([row], auxiliary=[extra])
        self.assertIsNone(result['records'][0]['ir'])
        self.assertTrue(any(i['reason'] == 'dual_note_pairing_unresolved' for i in result['records'][0]['unresolved']))
        event['notes'][0].update(fret=0, staff_position=7)
        with self.assertRaises(ValueError):
            assemble([row], auxiliary=[extra])

    def test_dual_unison_head_and_its_tie_apply_to_the_observed_note_group(self):
        events = [{'start': i * 1920, 'duration': 'h', 'status': 'normal',
                   'notes': [{'string': 1, 'fret': 0 if i == 0 else None},
                             {'string': 2, 'fret': 5 if i == 0 else None}],
                   'notation_notes': [{'staff_position': 7, 'accidental': None}]}
                  for i in range(2)]
        relation = dict(connection('tie', [0, 0, 0], [0, 1, 0]), representation='notation')
        row = measure(0, voices=[{'voice': 0, 'events': events}], relations=[relation])
        row['mode'] = 'both'
        values = dict(time='4/4', key=0, clef='G2', clef_octave=0,
            tuning=[64, 59, 55, 50, 45, 40], instrument_transpose=-12,
            instrument_transpose_explicit=False, capo=0)
        extra = {'state': {k: {'status': 'observed', 'value': v} for k, v in values.items()}}
        result = assemble([row], auxiliary=[extra])
        self.assertIsNotNone(result['records'][0]['ir'])
        notes = result['records'][0]['resolved']['voices'][0]['events'][1]['notes']
        self.assertEqual([n['fret'] for n in notes], [0, 5])
        self.assertTrue(all('tie' in n['effects'] for n in notes))
        self.assertEqual(result['relations'][0]['start_notes'], [[0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(result['relations'][0]['end_notes'], [[0, 0, 1, 0], [0, 0, 1, 1]])
        self.assertIsNone(result['relations'][0]['start'])
        events[1]['notes'] = []
        result = assemble([row], auxiliary=[extra])
        self.assertIsNone(result['records'][0]['ir'])
        hidden = result['records'][0]['resolved']['voices'][0]['events'][1]['notes']
        self.assertEqual(len(hidden), 1)
        self.assertNotIn('string', hidden[0])
        self.assertEqual(hidden[0]['tab_string_candidates'], [1, 2])

    def test_notation_ties_restore_hidden_tab_chord_without_hidden_supervision(self):
        events = [{'start': i * 960, 'duration': 'q', 'status': 'normal',
                   'notes': [{'string': 1, 'fret': 0}, {'string': 2, 'fret': 10}] if i == 0 else [],
                   'notation_notes': [{'staff_position': 10, 'accidental': None},
                                      {'staff_position': 7, 'accidental': None}]}
                  for i in range(4)]
        relations = [connection('tie', [0, i, n], [0, i+1, n], representation='notation')
                     for i in range(3) for n in range(2)]
        row = measure(0, voices=[{'voice': 0, 'events': events}], relations=relations)
        row['mode'] = 'both'
        values = dict(time='4/4', key=0, clef='G2', clef_octave=0,
            tuning=[64, 59, 55, 50, 45, 40], instrument_transpose=-12,
            instrument_transpose_explicit=False, capo=0)
        extra = {'state': {k: {'status': 'observed', 'value': v} for k, v in values.items()}}
        result = assemble([row], auxiliary=[extra])
        self.assertIsNotNone(result['records'][0]['ir'])
        self.assertEqual(result['records'][0]['observation'], row['observation'])
        for event in result['records'][0]['resolved']['voices'][0]['events']:
            self.assertEqual({(n['string'], n['fret'], n['pitch']) for n in event['notes']},
                             {(1, 0, 76), (2, 10, 81)})
        self.assertEqual(result['relations'][0]['start'], [0, 0, 0, 1])

    def test_percussion_position_and_head_are_both_required(self):
        row = measure(0)
        row['mode'] = 'notation'
        row['observation']['state']['clef'] = {'status': 'observed', 'value': 'percussion'}
        for event, (position, head) in zip(row['observation']['voices'][0]['events'],
                [(9, 'x'), (9, 'circle-x'), (8, 'diamond'), (1, 'normal')]):
            event['notes'] = [dict(staff_position=position, notehead=head)]
        result = assemble([row])
        self.assertEqual([note(result, 0, i)['pitch'] for i in range(4)], [42, 46, 53, 36])
        self.assertIsNotNone(result['records'][0]['ir'])
        row['instrument'] = 'drums'
        row['observation']['state']['clef'] = {'status': 'observed', 'value': 'C3'}
        from research.inference.run import assemble_predictions
        result = assemble_predictions([dict(row, id='drums')])
        self.assertEqual([note(result, 0, i)['pitch'] for i in range(4)], [42, 46, 53, 36])
        row['observation']['voices'][0]['events'][0]['notes'][0]['notehead'] = None
        self.assertIsNone(assemble([row])['records'][0]['ir'])
        from research.evaluation.measures import visible_equal
        self.assertFalse(visible_equal(dict(staff_position=9, notehead='x'),
            dict(staff_position=9, notehead='circle-x'), 'notation'))

    def test_dual_tied_accidental_does_not_create_a_false_natural_pitch(self):
        rows = [measure(0, (6, 6, 6, 6), [connection('tie', [0, 3, 0], 'right')]),
                measure(1, (6, 6, 6, 6), [connection('tie', 'left', [0, 0, 0])])]
        for row in rows:
            row['mode'] = 'both'
            for event in row['observation']['voices'][0]['events']:
                event['notes'][0]['string'] = 3
                event['notation_notes'] = [{'staff_position': 5, 'accidental': None}]
            for relation in row['observation']['relations']:
                relation['representation'] = 'notation'
        rows[0]['observation']['voices'][0]['events'][0]['notation_notes'][0]['accidental'] = 1
        values = dict(time='4/4', key=0, clef='G2', clef_octave=0,
            tuning=[64, 59, 55, 50, 45, 40], instrument_transpose=-12,
            instrument_transpose_explicit=False, capo=0)
        extra = {'state': {k: {'status': 'observed', 'value': v} for k, v in values.items()}}
        result = assemble(rows, auxiliary=[extra, extra])
        self.assertTrue(all(row['ir'] for row in result['records']))
        self.assertEqual([note(result, 1, i)['pitch'] for i in range(4)], [73] * 4)
        self.assertIsNone(result['records'][1]['observation']['voices'][0]['events'][1]['notation_notes'][0]['accidental'])

    def test_evaluation_keeps_unknown_rhythm_separate_without_reordering_repeated_notes(self):
        from research.evaluation.measures import summarize
        from research.inference.run import assemble_predictions, observation_record

        row = dict(measure(0, frets=(5, 5, 7)), id='unknown-rhythm', image='measure.png')
        expected = row['observation']['voices'][0]['events']
        expected[0]['duration'] = None
        expected[1].update(start=None, duration=None)
        expected[1]['notes'][0]['effects'] = ['accent']
        expected[2]['start'] = 960
        row['observation']['relations'] = [connection('pm', [0, 1], None)]

        def evaluate(observation):
            predictions = [observation_record(row, observation)]
            return summarize([row], predictions, assemble_predictions(predictions))

        perfect = evaluate(row['observation'])
        metrics = perfect['metrics']
        self.assertFalse(perfect['failures'])
        self.assertEqual(metrics['visible_notes']['f1'], 1)
        self.assertEqual(metrics['rhythm'], dict(expected=1, predicted=1, correct=1,
                                                precision=1, recall=1, f1=1))
        self.assertEqual(metrics['known_rhythm_fields']['start']['expected'], 2)
        self.assertEqual(metrics['known_rhythm_fields']['duration']['expected'], 1)
        self.assertEqual(metrics['unknown_rhythm']['start']['preserved_fraction'], 1)
        self.assertEqual(metrics['unknown_rhythm']['duration']['preserved'], 2)
        self.assertEqual(metrics['local_relations']['f1'], 1)
        self.assertEqual(metrics['visible_effects']['f1'], 1)

        invented = deepcopy(row['observation'])
        invented['voices'][0]['events'][1].update(start=960, duration='q')
        metrics = evaluate(invented)['metrics']
        self.assertEqual(metrics['visible_notes']['f1'], 1)
        self.assertEqual(metrics['rhythm']['f1'], 1)
        self.assertEqual(metrics['unknown_rhythm']['start']['invented'], 1)
        self.assertEqual(metrics['unknown_rhythm']['duration']['invented'], 1)

        # A wildcard onset would swap these notes and incorrectly score all
        # three as correct. The unknown note stays at its observed ordinal.
        swapped = deepcopy(invented)
        swapped['voices'][0]['events'][1]['notes'][0]['fret'] = 7
        swapped['voices'][0]['events'][2]['notes'][0]['fret'] = 5
        self.assertEqual(evaluate(swapped)['metrics']['visible_notes']['correct'], 1)

        missing = deepcopy(row['observation'])
        missing['voices'][0]['events'].pop(1)
        missing['relations'] = []
        metrics = evaluate(missing)['metrics']
        self.assertEqual(metrics['visible_notes']['correct'], 2)
        self.assertEqual(metrics['unknown_rhythm']['start']['matched'], 0)
        self.assertEqual(metrics['unknown_rhythm']['start']['missing'], 1)
        self.assertEqual(metrics['unknown_rhythm']['start']['preserved_fraction'], 0)

        # Known values remain independently scorable in a partial event.
        expected[1]['duration'] = 'e'
        wrong_duration = deepcopy(row['observation'])
        wrong_duration['voices'][0]['events'][1]['duration'] = 'q'
        metrics = evaluate(wrong_duration)['metrics']
        self.assertEqual(metrics['known_rhythm_fields']['duration']['expected'], 2)
        self.assertEqual(metrics['known_rhythm_fields']['duration']['correct'], 1)

    def test_native_single_slide_ink_keeps_scale_and_owner_uncertainty(self):
        system = {'measure_boxes': [{'measure_index': 0}]}
        owner = (0, 0, 0, 0, 0)
        for scale in (0.5, 1, 2):
            with self.subTest(scale=scale):
                page = EvidencePage()
                page.drawing.lines = [{'stroke': True, 'pts': [
                    (x * scale * 72 / 25.4, y * scale * 72 / 25.4)
                    for x, y in [(8, 20), (9, 19)]]}]
                grouped = {owner: [{'kind': 'fret', 'bbox_mm':
                                   [value * scale for value in (9.5, 18, 2, 3)]}]}
                crop = [value * scale for value in (0, 0, 30, 30)]
                effects, consumed, unknown = native_single_slides(
                    page, grouped, 0, system, 0, crop, set())
                self.assertEqual(effects, {owner: {'sib'}})
                self.assertEqual(consumed, {0})
                self.assertFalse(unknown)
                grouped[(0, 0, 1, 0, 0)] = deepcopy(grouped[owner])
                effects, consumed, unknown = native_single_slides(
                    page, grouped, 0, system, 0, crop, set())
                self.assertFalse(effects)
                self.assertEqual(consumed, {0})
                self.assertEqual(unknown, [connection(None, None, None, shape='diagonal')])
                grouped[(0, 0, 1, 0, 0)][0]['bbox_mm'] = [
                    value * scale for value in (9, 18, 3, 3)]
                effects, _, unknown = native_single_slides(
                    page, grouped, 0, system, 0, crop, set())
                self.assertFalse(effects)
                self.assertEqual(unknown, [connection(None, None, None, shape='diagonal')])
        # A longer non-square stroke with two visible contacts must survive
        # even in older exports that do not provide native relation tracing.
        page = EvidencePage()
        page.drawing.lines = [{'stroke': True, 'pts': [
            (x * 72 / 25.4, y * 72 / 25.4) for x, y in [(12, 20), (18, 19)]]}]
        grouped = {owner: [{'kind': 'fret', 'bbox_mm': [9.5, 18, 2, 3]}],
                   (0, 0, 0, 1, 0): [{'kind': 'fret', 'bbox_mm': [18.5, 18, 2, 3]}]}
        effects, consumed, unknown = native_single_slides(
            page, grouped, 0, system, 0, [0, 0, 30, 30], set())
        self.assertFalse(effects)
        self.assertEqual(consumed, {0})
        self.assertEqual(unknown, [connection(None, [0, 0, 0], [0, 1, 0], shape='diagonal')])

    @unittest.skipUnless(os.environ.get('GUITAROCR_NATIVE_COVERAGE'), 'Set GUITAROCR_NATIVE_COVERAGE to a native-coverage GP export directory')
    def test_rendered_gp_coverage_and_hidden_secondary_voice_rhythm(self):
        directory = Path(os.environ['GUITAROCR_NATIVE_COVERAGE'])
        layout = json.loads((directory / 'layout.json').read_text())
        score = json.loads((directory / 'official-score.json').read_text())
        with open_pdf(directory / 'score.pdf') as document:
            results = native_observations(layout, score, document)
            changed = deepcopy(score)
            for staff in changed['tracks'][0]['staves']:
                for bar in staff['measures']:
                    for voice in bar['voices']:
                        for event in voice['events']:
                            for item in event['notes']:
                                item['fret'] = (item['fret'] or 0) + 7
                                for field in ('tie_origin', 'tie_destination', 'hopo_origin',
                                              'hopo_destination', 'palm_muted', 'let_ring'):
                                    item[field] = not item.get(field, False)
                                for field in item.get('slides', {}):
                                    item['slides'][field] = not item['slides'][field]
            mutated = native_observations(layout, changed, document)
            self.assertEqual({index: row.get('observation') for index, row in results.items()},
                             {index: row.get('observation') for index, row in mutated.items()})
            nonstandard = deepcopy(score)
            for event in nonstandard['tracks'][0]['staves'][0]['measures'][17]['voices'][0]['events'][:3]:
                event['rhythm']['tuplets']['primary'] = [3, 4]
            unknown_ratio = native_observations(layout, nonstandard, document)
        self.assertEqual(len(results), 32)
        self.assertFalse({index: row['error'] for index, row in results.items() if 'error' in row})
        for row in results.values():
            parse_observation(row['observation'])
        kinds = {relation['kind'] for row in results.values() for relation in row['observation']['relations']}
        self.assertTrue({'tie', 'hammer', 'sl', 'ss', 'pm', 'let'} <= kinds)
        # GP prints the primary voice rhythm here but suppresses voice 1's
        # triplet bracket and subdivision stems. Source ratios are not ink.
        voices = {voice['voice']: voice['events'] for voice in results[19]['observation']['voices']}
        self.assertTrue(all(event['duration'] == 'q' for event in voices[0]))
        self.assertTrue(all(event['duration'] is None for event in voices[1][:3]))
        self.assertEqual(voices[1][0]['start'], 0)
        self.assertIsNone(voices[1][1]['start'])
        self.assertIsNone(voices[1][2]['start'])
        for index, kind in ((20, 'pm'), (21, 'let')):
            self.assertTrue(any(relation['kind'] == kind and relation['end'] is None
                                for relation in results[index]['observation']['relations']))
        effects = {effect for voice in results[14]['observation']['voices']
                   for event in voice['events'] for item in event['notes']
                   for effect in item.get('effects', [])}
        self.assertTrue({'sia', 'sib', 'sod', 'sou'} <= effects)
        for event in results[17]['observation']['voices'][0]['events'][:3]:
            self.assertIsNotNone(event['duration'])
        for event in unknown_ratio[17]['observation']['voices'][0]['events'][:3]:
            self.assertIsNone(event['duration'])

    def test_native_connections_use_drawn_paths_and_visible_frets_not_internal_names(self):
        factor = 72 / 25.4
        points = [[11, 20], [13, 18], [17, 18], [19, 20]]
        pdf_points = [tuple(value * factor for value in point) for point in points]
        curve = {'path': [('m', pdf_points[0]), ('c', *pdf_points[1:])], 'stroke': True}
        system = {'page': 1, 'system_index': 0, 'bbox_mm': [0, 0, 30, 30],
                  'measure_boxes': [{'measure_index': 0}]}
        start = {'staff_index': 0, 'measure_index': 0, 'voice_index': 0, 'event_index': 0,
                 'note_index': 0, 'native_string_index': 2, 'visibility_code': 0,
                 'render_origin_mm': [10, 20], 'layout_bbox_mm': [9, 19, 2, 2], 'page': 1}
        end = dict(start, event_index=1, render_origin_mm=[20, 20], layout_bbox_mm=[19, 19, 2, 2])
        for hidden_fret, destination in ((True, None), (False, '4'), (False, '5')):
            with self.subTest(hidden=hidden_fret, printed_fret=destination):
                end_owner = dict(end, visibility_code=int(hidden_fret), fret_text='(999)' if hidden_fret else destination)
                trace = {'view_kind': 'hopo_slur', 'page': 1, 'system_index': 0,
                         'points_mm': points, 'start_owner': start, 'end_owner': end_owner}
                layout = {'relation_geometry': {'status': 'complete', 'relations': [trace]}}
                grouped = {(0, 0, 0, 0, 0): [dict(start, bbox_mm=start['layout_bbox_mm'], kind='fret', text='4')]}
                if not hidden_fret:
                    grouped[0, 0, 0, 1, 0] = [dict(end, bbox_mm=end['layout_bbox_mm'], kind='fret', text=destination)]
                relations, effects, anchors, used, _, _ = native_connection_evidence(
                    EvidencePage(curves=[curve]), layout, grouped, 0, system, 0, (0, 0, 30, 30))
                self.assertEqual(relations[0]['kind'], 'tie' if hidden_fret or destination == '4' else None)
                self.assertFalse(effects)
                self.assertEqual(used, {0})
                if hidden_fret:
                    self.assertEqual(anchors[0, 0, 0, 1, 0]['text'], '')
                    self.assertNotIn('fret', anchors[0, 0, 0, 1, 0])
                    self.assertNotIn('fret_text', anchors[0, 0, 0, 1, 0])
        trace.update(view_kind='beat_slur', start_owner=None, end_owner=None)
        relations, _, _, used, _, _ = native_connection_evidence(
            EvidencePage(curves=[curve]), layout, grouped, 0, system, 0, (0, 0, 30, 30))
        self.assertEqual(used, {0})
        self.assertIsNone(relations[0]['start'])
        self.assertIsNone(relations[0]['end'])

    def test_slide_label_belongs_between_note_heads_when_digits_shorten_the_line(self):
        # Actual GP 8 -> 10 geometry: the wide destination shortens the slash;
        # its sl. text still labels the connection between the two note heads.
        factor = 72 / 25.4
        points = [[63.9705687061, 93.2080553877], [64.8681649178, 92.1280553448]]
        printed = (182.1550823076, 246.676033908, 190.0930823076, 252.676033908, 'sl.')
        line = {'stroke': True, 'pts': [tuple(value * factor for value in point) for point in points]}
        grouped = {(0, 0, 0, 0, 0): [{'kind': 'fret', 'text': '8', 'native_string_index': 4,
                    'bbox_mm': [61.8623742566, 91.1740414718, 1.35819444954, 2.988027789],
                    'render_origin_mm': [62.5414714813, 92.6680553663]}],
                   (0, 0, 0, 1, 0): [{'kind': 'fret', 'text': '10', 'native_string_index': 4,
                    'bbox_mm': [65.6181649178, 91.1740414718, 2.71638889909, 2.988027789],
                    'render_origin_mm': [66.9763593674, 92.6680553663]}]}
        system = {'page': 1, 'system_index': 0, 'bbox_mm': [60, 80, 10, 20],
                  'measure_boxes': [{'measure_index': 0}]}
        layout = {'relation_geometry': {'status': 'complete', 'relations': [
            {'page': 1, 'system_index': 0, 'points_mm': points, 'start_owner': None, 'end_owner': None}]}}
        relations, effects, _, _, consumed, labels = native_connection_evidence(
            EvidencePage(lines=[line], words=[printed]), layout, grouped, 0, system, 0, (60, 80, 70, 100))
        self.assertFalse(relations)
        self.assertEqual(effects, {(0, 0, 0, 0, 0): {'ss'}})
        self.assertEqual(consumed, {0})
        self.assertEqual(labels, [list(printed[:4])])

    def test_native_tuplet_requires_the_printed_number_and_its_bracket(self):
        # Coordinates from a real GP PDF: two drawn L paths surround the 3.
        halves = [[(483.6455916, 242.75779656), (466.81862069, 242.75779656),
                   (466.81862069, 240.63181224)],
                  [(491.41677288, 242.75779656), (508.24374384, 242.75779656),
                   (508.24374384, 240.63181224)]]
        system = {'measure_boxes': [{'measure_index': 68}]}
        grouped = {(0, 68, 0, event, 0): [glyph(x0, x1)] for event, (x0, x1) in enumerate([
            (462.9655, 470.6496), (483.6789, 491.3629), (504.3922, 512.0763),
            (525.1056, 532.7897)])}
        crop = [value * 25.4 / 72 for value in (450, 0, 550, 270)]
        for slope in (0, 0.04, -0.04):
            for complete in (True, False):
                with self.subTest(slope=slope, printed_number=complete):
                    curves = [{'stroke': True, 'path': [(command, (x, y + slope * (x - 488)))
                              for command, (x, y) in zip(('m', 'l', 'l'), half)]} for half in halves]
                    curves.append({'stroke': True, 'path': [('m', (470, 40)),
                        ('c', (480, 35), (495, 35), (505, 40))]})
                    chars = [{'text': '3', 'x0': 484.4584, 'x1': 492.0724,
                              'top': 238.6303, 'bottom': 247.6303}] if complete else []
                    counts, used, boxes = native_tuplet_evidence(
                        EvidencePage(curves=curves, chars=chars), grouped, 0, system, 68, crop)
                    self.assertEqual(counts, {(0, 68, 0, i): 3 for i in range(3)} if complete else {})
                    self.assertEqual(used, {0, 1} if complete else set())
                    self.assertEqual(len(boxes), int(complete))

    def test_connection_middle_does_not_copy_a_voice_from_outside_the_crop(self):
        factor = 72 / 25.4
        points = [[1, 20], [10, 18], [20, 18], [29, 20]]
        pdf_points = [tuple(value * factor for value in point) for point in points]
        curve = {'path': [('m', pdf_points[0]), ('c', *pdf_points[1:])], 'stroke': True}
        system = {'page': 1, 'system_index': 0, 'bbox_mm': [0, 0, 30, 30],
                  'measure_boxes': [{'measure_index': index} for index in range(3)]}
        grouped = {(0, 1, voice, 0, 0): [{'kind': 'fret', 'text': '4',
                   'native_string_index': voice, 'bbox_mm': [14, 19 + voice * 4, 2, 2],
                   'render_origin_mm': [15, 20 + voice * 4]}] for voice in (0, 1)}
        results = []
        for external_voice in (0, 1):
            start = {'staff_index': 0, 'measure_index': 0, 'voice_index': external_voice,
                     'event_index': 0, 'note_index': 0, 'visibility_code': 0}
            trace = {'page': 1, 'system_index': 0, 'points_mm': points,
                     'start_owner': start, 'end_owner': dict(start, measure_index=2)}
            layout = {'relation_geometry': {'status': 'complete', 'relations': [trace]}}
            relations, effects, _, _, _, _ = native_connection_evidence(
                EvidencePage(curves=[curve]), layout, grouped, 0, system, 1, (10, 0, 20, 30))
            self.assertEqual(relations, [connection(None, 'left', 'right', shape='curve')])
            self.assertFalse(effects)
            results.append(relations)
        self.assertEqual(results[0], results[1])

    def test_scope_voice_is_not_disambiguated_by_an_endpoint_outside_the_crop(self):
        system = {'measure_boxes': [{'measure_index': 0}, {'measure_index': 1}]}
        span = {'kind': 'pm', 'style': 'labelled_dash', 'text_bbox': [10, 20, 14, 25],
                'start_x': 12, 'end_x': 82, 'line_y': 22}
        page = EvidencePage(dashes=[(16, 22, 82)])
        for local_measure in (0, 1):
            with self.subTest(local_measure=local_measure):
                local_x, external_x = (10, 80) if local_measure == 0 else (80, 10)
                local = {(0, local_measure, voice, 0, 0): [glyph(local_x, local_x + 4)]
                         for voice in (0, 1)}
                crop = [value * 25.4 / 72 for value in
                        ((0, 0, 50, 60) if local_measure == 0 else (50, 0, 100, 60))]
                results = []
                for external_voice in (0, 1):
                    grouped = {**local, (0, 1 - local_measure, external_voice, 0, 0):
                               [glyph(external_x, external_x + 4)]}
                    relations, _, _ = native_scope_evidence(
                        page, [span], grouped, 0, system, local_measure, crop)
                    self.assertNotIn('voice', relations[0])
                    self.assertEqual(relations[0]['start'], None if local_measure == 0 else 'left')
                    self.assertEqual(relations[0]['end'], 'right' if local_measure == 0 else None)
                    results.append(relations)
                self.assertEqual(results[0], results[1])

    def test_scope_voice_is_not_selected_by_simultaneous_fret_text_width(self):
        system = {'measure_boxes': [{'measure_index': 0}]}
        span = {'kind': 'let', 'style': 'labelled_dash', 'text_bbox': [10, 20, 17, 25],
                'start_x': 13.5, 'end_x': 33.5, 'line_y': 22}
        page = EvidencePage(dashes=[(18, 22, 33.5)])
        crop = [value * 25.4 / 72 for value in (0, 0, 50, 60)]
        for wide_start in (0, 1):
            for wide_end in (0, 1):
                with self.subTest(wide_start=wide_start, wide_end=wide_end):
                    grouped = {}
                    for event, (origin, wide_voice) in enumerate(((12, wide_start), (32, wide_end))):
                        for voice in (0, 1):
                            half_width = 2 if voice == wide_voice else 1
                            grouped[(0, 0, voice, event, 0)] = [
                                glyph(origin - half_width, origin + half_width, y=40 + 6 * voice)]
                    relations, _, _ = native_scope_evidence(
                        page, [span], grouped, 0, system, 0, crop)
                    self.assertEqual(relations, [connection('let', None, None)])

    def test_native_scopes_keep_partial_and_ambiguous_visible_evidence(self):
        system = {'measure_boxes': [{'measure_index': 0}, {'measure_index': 1}]}
        grouped = {(0, m, 0, event, 0): [glyph(x, x + 4, y=40 - 5 * string, string=string)]
                   for m, event, x, string in [(0, 0, 10, 0), (0, 1, 30, 1),
                                               (1, 0, 60, 0), (1, 1, 80, 1)]}
        crop = [value * 25.4 / 72 for value in (0, 0, 50, 60)]
        label = {'kind': 'pm', 'style': 'label_only', 'text_bbox': [10, 20, 14, 25],
                 'start_x': 12, 'end_x': 14, 'line_y': 22}
        relations, _, _ = native_scope_evidence(
            EvidencePage(), [label], grouped, 0, system, 0, crop, 6)
        self.assertEqual(relations[0]['start'], [0, 0])
        self.assertIsNone(relations[0]['end'])
        pm = dict(label, style='labelled_dash', end_x=32)
        let = dict(pm, kind='let', text_bbox=[10, 10, 14, 15], line_y=12)
        page = EvidencePage(dashes=[(14, 22, 32), (14, 12, 32)])
        relations, _, _ = native_scope_evidence(page, [pm, let], grouped, 0, system, 0, crop, 6)
        self.assertEqual(len(relations), 2)
        self.assertGreater(relations[0]['lane'], relations[1]['lane'])
        two_voices = {**grouped, **{(a, b, 1, d, e): gs for (a, b, _, d, e), gs in grouped.items()}}
        relations, _, _ = native_scope_evidence(page, [pm], two_voices, 0, system, 0, crop, 6)
        self.assertEqual(relations[0]['kind'], 'pm')
        self.assertIsNone(relations[0]['start'])
        self.assertIsNone(relations[0]['end'])
        self.assertIsNone(relations[0].get('voice'))
        continuation = dict(pm, end_x=82)
        relations, _, _ = native_scope_evidence(
            EvidencePage(dashes=[(14, 22, 82)]), [continuation], grouped, 0, system, 1,
            [value * 25.4 / 72 for value in (50, 0, 100, 60)], 6)
        self.assertIsNone(relations[0]['kind'])
        self.assertEqual((relations[0]['start'], relations[0]['end']), ('left', [0, 1]))

    def test_tied_hidden_fret_is_filled_only_after_joining_adjacent_measures(self):
        rows = [measure(0, relations=[connection('tie', [0, 3, 0], 'right')]),
                measure(1, (None, 3, 3, 3), [connection('tie', 'left', [0, 0, 0])])]
        original = deepcopy(rows)
        result = assemble(rows)
        self.assertEqual(rows, original)
        self.assertIsNone(result['records'][1]['observation']['voices'][0]['events'][0]['notes'][0]['fret'])
        self.assertEqual(note(result, 1, 0)['fret'], 3)
        self.assertIn('tie', note(result, 1, 0)['effects'])
        self.assertEqual(result['relations'][0]['status'], 'resolved')
        self.assertIsNotNone(result['records'][1]['ir'])

    def test_hammer_and_slides_join_without_copying_the_destination_fret(self):
        for kind, shape in [('hammer', 'curve'), ('sl', 'diagonal'), ('ss', 'diagonal')]:
            for target in (4, 2, None):
                with self.subTest(kind=kind, destination=target):
                    rows = [measure(0, relations=[connection(kind, [0, 3, 0], 'right', shape=shape)]),
                            measure(1, (target, 3, 3, 3),
                                    [connection(kind, 'left', [0, 0, 0], shape=shape)])]
                    result = assemble(rows)
                    self.assertEqual(note(result, 1, 0)['fret'], target)
                    self.assertIn(kind, note(result, 0, 3)['effects'])
                    self.assertEqual(result['relations'][0]['status'], 'resolved')
                    self.assertEqual(result['records'][1]['ir'] is None, target is None)

    def test_visible_tie_fret_conflict_is_preserved(self):
        result = assemble([
            measure(0, relations=[connection('tie', [0, 3, 0], 'right')]),
            measure(1, (4, 3, 3, 3), [connection('tie', 'left', [0, 0, 0])])])
        self.assertEqual(note(result, 1, 0)['fret'], 4)
        self.assertEqual(result['relations'][0]['reason'], 'tie_endpoint_conflict')
        self.assertNotIn('tie', note(result, 1, 0).get('effects', []))
        self.assertIsNotNone(result['records'][1]['ir'])

    def test_gp_slide_can_end_at_next_system_without_a_repeated_incoming_stroke(self):
        for kind in ('sl', 'ss'):
            with self.subTest(kind=kind):
                rows = [measure(0, relations=[connection(kind, [0, 3, 0], 'right', shape='diagonal')]),
                        measure(1, (7, 3, 3, 3), system=1)]
                result = assemble(rows)
                self.assertEqual(result['relations'][0]['status'], 'resolved')
                self.assertEqual(result['relations'][0]['end'], [1, 0, 0, 0])
                self.assertEqual(result['relations'][0]['end_evidence'], 'unique_adjacent_visible_note')
                self.assertEqual(result['records'][1]['observation']['relations'], [])
                self.assertEqual(note(result, 1, 0)['fret'], 7)
                self.assertIn(kind, note(result, 0, 3)['effects'])
                # Two competing notes must not turn a source-only mark into
                # a guessed destination; its observed origin effect survives.
                rows[1]['observation']['voices'][0]['events'][0]['notes'].append({'string': 1, 'fret': 9})
                ambiguous = assemble(rows)
                self.assertEqual(ambiguous['relations'][0]['end'], 'right')
                self.assertEqual(ambiguous['relations'][0]['status'], 'open')
                self.assertIn(kind, note(ambiguous, 0, 3)['effects'])

    def test_legato_slide_can_continue_as_only_its_curve(self):
        rows = [measure(0, relations=[connection('sl', [0, 3, 0], 'right', shape='diagonal')]),
                measure(1, (7, 3, 3, 3), [connection(None, 'left', [0, 0, 0], shape='curve')], system=1)]
        result = assemble(rows)
        self.assertEqual(len(result['relations']), 1)
        self.assertEqual(result['relations'][0]['status'], 'resolved')
        self.assertIn('sl', note(result, 0, 3)['effects'])
        self.assertEqual(note(result, 1, 0)['fret'], 7)

    def test_missing_measure_cannot_supply_a_hidden_fret(self):
        result = assemble([
            measure(0, relations=[connection('tie', [0, 3, 0], 'right')]),
            measure(2, (None, 3, 3, 3), [connection('tie', 'left', [0, 0, 0])])])
        self.assertIsNone(note(result, 1, 0)['fret'])
        self.assertIsNone(result['records'][1]['ir'])
        self.assertTrue(all(len(relation['segments']) == 1 for relation in result['relations']))

    def test_unknown_curve_through_a_crop_never_becomes_a_technique_scope(self):
        rows = [measure(0, relations=[connection(None, [0, 3, 0], 'right', shape='curve')]),
                measure(1, relations=[connection(None, 'left', 'right', shape='curve', voice=0)]),
                measure(2, relations=[connection(None, 'left', [0, 0, 0], shape='curve')])]
        result = assemble(rows)
        self.assertTrue(all(relation['status'] == 'unknown' for relation in result['relations']))
        self.assertTrue(all(not relation['coverage'] for relation in result['relations']))
        self.assertTrue(all(not row['resolved']['visual_techniques'] for row in result['records']))
        self.assertTrue(all(not note(result, r, e).get('effects') for r in range(3) for e in range(4)))

    def test_parallel_pm_and_let_use_order_only_with_continuous_layout(self):
        for same_system in (True, False):
            with self.subTest(same_system=same_system):
                rows = [measure(0, relations=[
                    connection('pm', [0, 0], 'right', lane=-0.2),
                    connection('let', [0, 1], 'right', lane=0.2)]),
                    measure(1, relations=[
                        connection(None, 'left', [0, 2], lane=20.0, shape='extension'),
                        connection(None, 'left', [0, 1], lane=10.0, shape='extension')],
                        system=0 if same_system else 1)]
                result = assemble(rows)
                if same_system:
                    self.assertEqual(len(result['relations']), 2)
                    self.assertEqual(set(note(result, 1, 0)['effects']), {'pm', 'let'})
                    self.assertEqual(note(result, 1, 2)['effects'], ['let'])
                    self.assertTrue(all(relation['status'] == 'resolved' for relation in result['relations']))
                else:
                    self.assertEqual(len(result['relations']), 4)
                    self.assertFalse(note(result, 1, 0).get('effects'))
                    self.assertEqual(sum(relation['status'] == 'unknown' for relation in result['relations']), 2)

    def test_label_only_and_unknown_voice_remain_visible_without_guessed_coverage(self):
        voices = measure(0)['observation']['voices']
        voices.append({'voice': 1, 'events': deepcopy(voices[0]['events'])})
        rows = [measure(0, voices=voices, relations=[
            connection('pm', None, None, shape='extension'),
            connection('let', [0, 1], None)])]
        result = assemble(rows)
        self.assertEqual(result['records'][0]['observation']['relations'], rows[0]['observation']['relations'])
        self.assertTrue(all(relation['status'] == 'unresolved' for relation in result['relations']))
        self.assertEqual(result['records'][0]['resolved']['visual_techniques'],
                         [{'voice': 0, 'event': 1, 'effect': 'let', 'relation': 1}])
        self.assertEqual(note(result, 0, 1, 0)['effects'], ['let'])
        self.assertTrue(all(not note(result, 0, e, v).get('effects')
                            for v in (0, 1) for e in range(4) if (v, e) != (0, 1)))

    def test_bad_local_pointer_does_not_block_a_correct_boundary_connection(self):
        rows = [measure(0, relations=[
            connection('tie', [0, 99, 0], 'right'), connection('tie', [0, 3, 0], 'right')]),
            measure(1, (None, 3, 3, 3), [connection('tie', 'left', [0, 0, 0])])]
        result = assemble(rows)
        self.assertEqual(note(result, 1, 0)['fret'], 3)
        self.assertEqual(sum(relation['status'] == 'resolved' for relation in result['relations']), 1)
        self.assertTrue(any(issue['reason'] == 'tie_endpoint_unresolved'
                            for issue in result['records'][0]['unresolved']))

    def test_relation_contract_accepts_visible_unknowns_and_note_connections(self):
        for kind, shape in [(None, 'curve'), (None, 'diagonal'), (None, 'extension'),
                            ('hammer', 'curve'), ('sl', 'diagonal'), ('ss', 'diagonal')]:
            with self.subTest(kind=kind, shape=shape):
                observation = measure(0, relations=[connection(kind, 'left', 'right', shape=shape)])['observation']
                self.assertEqual(parse_observation(observation), observation)


if __name__ == '__main__':
    unittest.main()
