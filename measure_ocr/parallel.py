"""Recognize a whole score using printed state and independent visual context."""

import json
from collections import Counter, defaultdict
from pathlib import Path
import time

from PIL import Image

from shared.constraints import gp5_timing_errors, validate_measure_target
from shared.glm_backend import create_backend
from shared.m2 import full_measure_rest_target
from shared.pitch_context import convert_pitch_target
from shared.score_state import SIGNATURE_PROMPT, attach_neighbours, measure_messages, parse_signature, resolve_fretted_pitches, resolve_measure_state, resolve_ties, signature_target
from shared.tasks import Cancelled


def _check(cancelled):
    if cancelled and cancelled():
        raise Cancelled("识别已停止，已完成的小节已保存，可以继续")


def _batch(backend, messages, tokens, grammars=None):
    if hasattr(backend, 'generate_batch'):
        return backend.generate_batch(messages, tokens, **({'grammar': grammars} if grammars else {}))
    return [backend.generate(m, tokens) for m in messages]


def read_score_states(records, backend, *, batch_size=24, cancelled=None, saved=None, reader=None):
    predictions = dict(saved or {})
    pending = [r for r in records if str(r['measure_number']) not in predictions]
    if reader is not None:
        for row, value in zip(pending, reader.predict(pending, cancelled=cancelled), strict=True):
            predictions[str(row['measure_number'])] = {
                **value, 'raw': signature_target(value['time'], value['key']),
                'tokens': 0, 'source': 'visual_classifier',
            }
        pending = []
    for offset in range(0, len(pending), batch_size):
        _check(cancelled)
        batch = pending[offset:offset + batch_size]
        messages = [[{'role': 'user', 'content': [
            {'type': 'image', 'url': row['image']}, {'type': 'text', 'text': SIGNATURE_PROMPT}
        ]}] for row in batch]
        for row, (text, tokens) in zip(batch, _batch(backend, messages, 40), strict=True):
            value = {'raw': text, 'tokens': tokens}
            try:
                value.update(parse_signature(text))
            except ValueError as error:
                value['error'] = str(error)
            predictions[str(row['measure_number'])] = value
    shared_times = {}
    if len({(r.get('part_id'), r.get('staff_id')) for r in records}) > 1 and all('bar_index' in r for r in records):
        observations = defaultdict(Counter)
        for row in records:
            if printed := predictions[str(row['measure_number'])].get('time'):
                observations[row['bar_index']][printed] += 1
        current = None
        for index in sorted({r['bar_index'] for r in records}):
            votes = observations[index].most_common()
            if votes and (len(votes) == 1 or votes[0][1] > votes[1][1]):
                current = votes[0][0]
            elif votes:
                current = None
            if current:
                shared_times[index] = current
    states = {}
    for row in records:
        part = (row.get('part_id', 'part-1'), row.get('staff_id', 'staff-1'))
        state, uncertain = states.setdefault(part, ({'time': '4/4', 'key': 0}, set()))
        prediction = predictions[str(row['measure_number'])]
        if prediction.get('error'):
            uncertain.update(('time', 'key'))
        for field in ('time', 'key'):
            if prediction.get(field) is not None:
                state[field] = prediction[field]
                uncertain.discard(field)
        if row.get('bar_index') in shared_times:
            state['time'] = shared_times[row['bar_index']]
            uncertain.discard('time')
        row['score_state'] = dict(state)
        row['state_needs_review'] = bool(uncertain)
        row['printed_state'] = prediction
    return predictions


def recognize_independent(
    records, mode, model_path, adapter_path, device, max_new_tokens,
    max_new_tokens_ceiling, tuning, maximum_attempts, diagnostics_path, resume,
    backend=None, progress=None, initial_records=None, retry_measures=None,
    cancelled=None, instrument='guitar', batch_size=8,
    state_reader=None,
    batch_order=None,
    constrained_decoding=None,
):
    diagnostics_path = Path(diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = {}
    if not resume:
        diagnostics_path.unlink(missing_ok=True)
    elif diagnostics_path.exists():
        # Recover a partially written final line without losing completed bars.
        offset = 0
        lines = diagnostics_path.read_bytes().splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if i != len(lines) - 1:
                    raise ValueError('Recognition journal contains an invalid record') from None
                with diagnostics_path.open('r+b') as h:
                    h.truncate(offset)
                break
            if row.get('accepted'):
                accepted[int(row['measure_number'])] = row
            offset += len(line)
        if diagnostics_path.stat().st_size and not diagnostics_path.read_bytes().endswith(b'\n'):
            with diagnostics_path.open('ab') as h:
                h.write(b'\n')
    retry = set(retry_measures or [])
    seeds = {int(r['measure_number']): r for r in (initial_records or []) if int(r['measure_number']) not in retry}
    numbers = {int(r['measure_number']) for r in records}
    if not retry.issubset(numbers):
        raise ValueError('Invalid measure number to retry')
    for row in records:
        row['mode'] = row.get('mode') or mode
        row['instrument'] = row.get('instrument') or instrument
        row['tuning'] = row.get('tuning', tuning)
    attach_neighbours(records)
    backend = backend or create_backend(model_path, adapter_path, device)
    capabilities_path = Path(adapter_path or model_path) / 'capabilities.json'
    capabilities = json.loads(capabilities_path.read_text()) if capabilities_path.exists() else {}
    constrained = capabilities.get('m2_constraints', False) if constrained_decoding is None else constrained_decoding
    constrained = constrained and getattr(backend, 'supports_json_schema', False)
    retry_constrained = (constrained_decoding is not False and capabilities.get('m2_retry_constraints', False)
                         and getattr(backend, 'supports_json_schema', False))
    for row in records:
        row['visual_pitch'] = bool(capabilities.get('visual_pitch', row.get('visual_pitch', False)))
    if state_reader is None:
        capabilities_path = Path(adapter_path or model_path) / 'capabilities.json'
        capabilities = json.loads(capabilities_path.read_text()) if capabilities_path.exists() else {}
        if capabilities.get('state_reader'):
            from measure_ocr.state_reader import cached_reader, state_model_path

            state_model = state_model_path(capabilities_path.parent / capabilities['state_reader']).resolve()
            state_reader = cached_reader(str(state_model), device, state_model.stat().st_mtime_ns)
    started = time.perf_counter()
    state_path = diagnostics_path.with_name('state_predictions.json')
    saved_states = json.loads(state_path.read_text()) if resume and state_path.exists() else None
    states = read_score_states(records, backend, cancelled=cancelled, saved=saved_states, reader=state_reader)
    state_path.write_text(json.dumps(states, ensure_ascii=False))
    state_seconds = time.perf_counter() - started
    pending, completed = [], 0
    for row in records:
        number = int(row['measure_number'])
        saved = seeds.get(number) or (accepted.get(number) if number not in retry else None)
        if saved:
            target = saved['target']
            row['tuning'] = saved.get('tuning', row['tuning'])
            visual_tuning = row['mode'] == 'both' and row.get('visual_pitch')
            _, errors = validate_measure_target(target, row['mode'], tuning=None if visual_tuning else row['tuning'], string_count=len(row['tuning']))
            if errors:
                raise ValueError(f'Invalid saved measure {number}: {errors}')
            row.update({k: v for k, v in saved.items() if k in {
                'target', 'written_target', 'manually_edited', 'needs_review', 'fallback_reason', 'recognition_attempts', 'boundary_resolved', 'tuning', 'tuning_source', 'tuning_explicit'
            }})
            row['recognition_attempts'] = saved.get('attempt', saved.get('recognition_attempts', 0))
            row['needs_review'] = (bool(saved.get('needs_review')) if saved.get('manually_edited') else
                                   bool(row.get('needs_review') or saved.get('fallback_reason') or
                                        row.get('pitch_needs_review') or row.get('state_needs_review')))
            completed += 1
        else:
            with Image.open(row['image']) as image:
                area = image.width * image.height
            pending.append((area, row))
    if progress:
        progress(completed, len(records))
    # Adjacent requests share their three-image neighbourhood in vLLM's
    # encoder cache. Padded generation instead benefits from similar sizes.
    if batch_order == 'area' or (batch_order is None and not getattr(backend, 'supports_ragged_batch', False)):
        pending.sort(key=lambda item: item[0])
    decode_start = time.perf_counter()
    with diagnostics_path.open('a') as journal:
        def save(value):
            journal.write(json.dumps(value, ensure_ascii=False) + '\n')
            journal.flush()

        for offset in range(0, len(pending), batch_size):
            batch = [r for _, r in pending[offset:offset + batch_size]]
            messages = [measure_messages(row) for row in batch]
            budgets = [max_new_tokens] * len(batch)
            unresolved = list(range(len(batch)))
            for attempt in range(1, maximum_attempts + 1):
                _check(cancelled)
                token_budget = max(budgets[i] for i in unresolved)
                grammars = None
                if constrained or (retry_constrained and attempt > 1):
                    from shared.m2_grammar import measure_grammar

                    grammars = [measure_grammar(batch[i]['mode'], len(batch[i]['tuning']) or 12) for i in unresolved]
                outputs = _batch(backend, [messages[i] for i in unresolved], token_budget, grammars)
                next_unresolved = []
                for i, (raw, tokens) in zip(unresolved, outputs, strict=True):
                    from measure_ocr.recognizer import _repair_truncated_optional_text, _retry_error_text

                    row = batch[i]
                    text = raw.strip()
                    text = text[text.find('M2'):] if 'M2' in text else text
                    text, repairs = _repair_truncated_optional_text(text)
                    try:
                        text, repaired = resolve_measure_state(text, row)
                        if repaired:
                            repairs.append('align_printed_signatures')
                    except (ValueError, KeyError, TypeError):
                        pass
                    target, conversion_errors = text, []
                    if row['mode'] == 'both' and row.get('tuning') and not row.get('visual_pitch'):
                        try:
                            target, repaired = resolve_fretted_pitches(text, row['tuning'])
                            if repaired:
                                repairs.append('derive_pitch_from_fingering')
                        except (ValueError, KeyError, TypeError):
                            # Syntax failures retain the normal structural retry.
                            pass
                    if (row['mode'] == 'notation' or (row['mode'] == 'both' and row.get('visual_pitch'))) and row['instrument'] != 'drums' and row.get('pitch_context'):
                        try:
                            target = convert_pitch_target(text, row['pitch_context'], mode=row['mode'])
                        except (ValueError, KeyError, TypeError) as error:
                            conversion_errors.append(str(error))
                    visual_tuning = row['mode'] == 'both' and row.get('visual_pitch')
                    parsed, errors = validate_measure_target(target, row['mode'], tuning=None if visual_tuning else row['tuning'], string_count=len(row['tuning']))
                    errors += conversion_errors
                    structural_errors = list(errors)
                    if not errors and gp5_timing_errors(target):
                        errors.append('Events overlap within a voice. Re-read durations and onsets: each event must end no later than the next starts; simultaneous notes belong to a chord or another voice')
                    if not errors and row['mode'] == 'notation' and row['instrument'] in {'guitar', 'bass'} and row['tuning'] and row.get('tuning_explicit'):
                        from gp5_export.fingering import notation_fingering_errors

                        candidates = row.get('fingering_tunings') or [row['tuning']]
                        failures = [notation_fingering_errors(parsed, tuning) for tuning in candidates]
                        if all(failures):
                            errors += ['Check written pitches after transposition: ' + e for e in min(failures, key=len)]
                    hit_limit = tokens >= token_budget
                    if hit_limit and repairs and attempt < maximum_attempts:
                        errors.append('Optional text reached the output limit')
                    value = {
                        'measure_number': row['measure_number'], 'mode': row['mode'], 'image': row['image'],
                        'attempt': attempt, 'raw': raw, 'target': target, 'accepted': not errors,
                        'generated_token_count': tokens, 'token_budget': token_budget,
                        'hit_token_limit': hit_limit, 'constraint_errors': errors, 'deterministic_repairs': repairs,
                        'constrained_decoding': grammars is not None,
                        'score_state': row['score_state'],
                        'written_target': text, 'tuning': row['tuning'],
                    }
                    save(value)
                    if errors and attempt < maximum_attempts:
                        # Keep the original images and most recent draft. A
                        # third attempt otherwise repeats two drafts plus
                        # parser exceptions quoting both, overflowing context.
                        if attempt > 1:
                            messages[i] = messages[i][:1]
                        messages[i].extend([
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': text}]},
                            {'role': 'user', 'content': [{'type': 'text', 'text': 'Correct the FIRST measure only. ' + _retry_error_text(errors, compact=attempt > 1) + '. Return one complete M2 fragment.'}]},
                        ])
                        if hit_limit:
                            budgets[i] = min(max_new_tokens_ceiling, budgets[i] * 2)
                        next_unresolved.append(i)
                        continue
                    if structural_errors:
                        numerator, denominator = map(int, row['score_state']['time'].split('/'))
                        target = full_measure_rest_target((numerator, denominator))
                        save({**value, 'target': target, 'accepted': True, 'fallback_reason': errors,
                              'attempt': attempt + 1, 'deterministic_repairs': ['fallback_full_measure_rest']})
                    elif errors:
                        save({**value, 'accepted': True, 'needs_review': True, 'fallback_reason': errors})
                    row.update(target=target, written_target=text, recognition_attempts=attempt, fallback_reason=errors,
                               needs_review=bool(errors or row.get('pitch_needs_review') or row.get('state_needs_review')))
                    completed += 1
                    if progress:
                        progress(completed, len(records))
                unresolved = next_unresolved
                if not unresolved:
                    break
        before = [r['target'] for r in records]
        from shared.score_tuning import reconcile_score_tuning

        reconcile_score_tuning(records)
        resolve_ties(records)
        for old, row in zip(before, records, strict=True):
            if old != row['target'] or row.get('tuning_source'):
                save({'measure_number': row['measure_number'], 'target': row['target'], 'accepted': True,
                      'attempt': row.get('recognition_attempts', 1), 'needs_review': row.get('needs_review', False),
                      'tuning': row['tuning'], 'tuning_source': row.get('tuning_source'),
                      'written_target': row.get('written_target'),
                      'manually_edited': row.get('manually_edited', False),
                      'tuning_explicit': row.get('tuning_explicit', False),
                      'fallback_reason': row.get('fallback_reason', []), 'deterministic_repairs': ['resolve_cross_bar_tie']})
    diagnostics_path.with_name('timing.json').write_text(json.dumps({
        'measures': len(records), 'state_seconds': state_seconds,
        'decode_seconds': time.perf_counter() - decode_start, 'total_seconds': time.perf_counter() - started,
        'batch_size': batch_size,
    }, indent=2))
    return [r['target'] for r in records]
