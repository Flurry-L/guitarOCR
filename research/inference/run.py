"""Read research observations, or run the application's native baseline."""

import argparse
from contextlib import ExitStack
from itertools import zip_longest
import json
from pathlib import Path
import time

import httpx

IDENTITY = ("source_id", "part_id", "staff_id", "bar_index", "mode")


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_rows(rows, limit=0):
    """Alternate modes while selecting whole documents and every observed staff."""
    if limit < 0:
        raise ValueError("Limit must be nonnegative")
    documents = {}
    for row in rows:
        documents.setdefault('drums' if row.get('instrument') == 'drums' else row['mode'], {}).setdefault(row.get('family', row['source_id']), []).append(row)
    selected = []
    for round_robin in zip_longest(*(mode.values() for mode in documents.values())):
        for document in round_robin:
            if document is None:
                continue
            selected.extend(document)
            if limit and len(selected) >= limit:
                return selected
    return selected


def base_messages(row):
    from research.data.training_samples import visual_measure_inputs

    sample = visual_measure_inputs(row)
    if len(sample['images']) != 1:
        raise ValueError("Observation recognition requires exactly one measure image")
    prompt = sample['messages'][0]['content'].replace('<image>', '')
    return [{'role': 'user', 'content': [
        {'type': 'image', 'url': sample['images'][0]}, {'type': 'text', 'text': prompt}]}]


def observation_record(row, raw, tokens=None):
    from research.inference.assembly import LAYOUT_FIELDS, parse_observation

    record = {key: row[key] for key in ('id', *IDENTITY)}
    record.update({key: row[key] for key in LAYOUT_FIELDS if key in row})
    for key in ('family', 'renderer', 'instrument'):
        if key in row:
            record[key] = row[key]
    if 'image' in row:
        record['image'] = row['image']
    record.update(raw=raw, tokens=tokens)
    try:
        record['observation'] = parse_observation(raw, row['mode'])
    except (ValueError, TypeError, KeyError) as error:
        record.update(observation=None, error=str(error))
    return record


def predict_observations(rows, model, device='cuda:0', batch_size=16, max_tokens=4096,
                         *, gpu_memory_utilization=.40):
    from research.inference.assembly import observation_schema
    from research.inference.vllm import VllmBackend

    if batch_size < 1 or max_tokens < 1:
        raise ValueError("Batch size and token budget must be positive")
    schemas = {key: observation_schema(*key)
               for key in {(row['mode'], row.get('instrument')) for row in rows}}
    backend = VllmBackend(model, device, options={
        'max_num_seqs': batch_size, 'max_model_len': 8192,
        'limit_mm_per_prompt': {'image': 1, 'video': 0},
        'gpu_memory_utilization': gpu_memory_utilization,
    })
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            outputs = [None] * len(batch)
            for key in dict.fromkeys((row['mode'], row.get('instrument')) for row in batch):
                indices = [i for i, row in enumerate(batch) if (row['mode'], row.get('instrument')) == key]
                generated = backend.generate_batch([base_messages(batch[i]) for i in indices], max_tokens,
                                                   schema=schemas[key])
                for index, output in zip(indices, generated, strict=True):
                    outputs[index] = output
            for row, (raw, tokens) in zip(batch, outputs, strict=True):
                yield observation_record(row, raw, tokens)
    finally:
        backend.close()


def read_auxiliary(path, rows):
    values = {row['id']: row for row in read_rows(path)}
    return [{**{key: values.get(row['id'], {}).get(key, {}) for key in ('state', 'voice_map')},
             'staff_start': values.get(row['id'], {}).get('staff_start', False)} for row in rows]


def assemble_predictions(predictions, auxiliary=None):
    from research.inference.assembly import LAYOUT_FIELDS, assemble

    indexes = [i for i, row in enumerate(predictions) if row.get('observation') is not None]
    inputs = [{**{key: predictions[i][key] for key in (*IDENTITY, 'observation')},
               **{key: predictions[i][key] for key in ('instrument',) if key in predictions[i]},
               **{key: predictions[i][key] for key in LAYOUT_FIELDS if key in predictions[i]}}
              for i in indexes]
    accepted = [auxiliary[i] for i in indexes] if auxiliary is not None else None
    assembled = assemble(inputs, auxiliary=accepted)
    by_index = dict(zip(indexes, assembled['records'], strict=True))
    records = []
    for index, prediction in enumerate(predictions):
        row = by_index.get(index)
        records.append({**prediction, **(row or {
            'resolved': None, 'ir': None, 'unresolved': ['invalid_observation'],
        })})
    # The assembler indexes accepted inputs; reports index all observations,
    # including malformed JSON that remains a failed measure.
    relations = []
    for relation in assembled['relations']:
        relation = dict(relation)
        for field in ('start', 'end'):
            if isinstance(relation.get(field), list):
                relation[field] = [indexes[relation[field][0]], *relation[field][1:]]
            if field + '_notes' in relation:
                relation[field + '_notes'] = [[indexes[item[0]], *item[1:]]
                                              for item in relation[field + '_notes']]
        relation['segments'] = [{**segment, 'record': indexes[segment['record']]}
                                for segment in relation.get('segments', [])]
        relation['coverage'] = [[indexes[point[0]], *point[1:]] for point in relation.get('coverage', [])]
        if 'voice_links' in relation:
            relation['voice_links'] = [{
                'from': [indexes[link['from'][0]], link['from'][1]],
                'to': [indexes[link['to'][0]], link['to'][1]],
                'evidence': [[indexes[point[0]], point[1]] for point in link['evidence']],
            } for link in relation['voice_links']]
        relations.append(relation)
    return {'records': records, 'relations': relations}


class NativeClient:
    def __init__(self, endpoint="http://127.0.0.1:7860"):
        self.http = httpx.Client(base_url=endpoint.rstrip("/"), timeout=120)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.http.close()

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"{method} {path}: {response.status_code} {response.text}")
        return response.json()

    def wait(self, sid):
        try:
            while True:
                project = self.request("GET", f"/api/sessions/{sid}")
                job = project.get("job") or {}
                if job.get("status") not in {"queued", "running"}:
                    if job.get("status") in {"failed", "cancelled"}:
                        raise RuntimeError(job.get("error") or job.get("message") or job["status"])
                    return project
                time.sleep(.5)
        except KeyboardInterrupt:
            self.request("POST", f"/api/sessions/{sid}/cancel")
            raise

    def recognize(self, inputs, output, *, mode="auto", session=None, action="process"):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        detect_after_import = session is None and action == "detect"
        if session:
            project = self.request("GET", f"/api/sessions/{session}")
            self.request("POST", f"/api/sessions/{session}/{action}", json={"mode": mode},
                         headers={"If-Match": str(project["revision"])})
        else:
            with ExitStack() as files:
                upload = [("files", (Path(path).name, files.enter_context(Path(path).open("rb"))))
                          for path in inputs]
                session = self.request("POST", "/api/sessions", files=upload,
                                       data={"action": "import" if detect_after_import else "full",
                                             "mode": mode})["id"]
        # The backend owns the project and its resume journal. Retain its ID even
        # when recognition is interrupted so the app or CLI can continue it.
        (output / "project.json").write_text(json.dumps({
            "endpoint": str(self.http.base_url), "id": session,
        }, indent=2) + "\n")
        project = self.wait(session)
        if detect_after_import:
            self.request("POST", f"/api/sessions/{session}/detect", json={"mode": mode},
                         headers={"If-Match": str(project["revision"])})
            project = self.wait(session)
        project["seconds"] = time.monotonic() - started
        for key, name in [("score_text_url", "score.txt"),
                          ("score_document_url", "score.json"),
                          ("musicxml_url", "score.musicxml")]:
            if project.get(key):
                response = self.http.get(project[key])
                response.raise_for_status()
                (output / name).write_bytes(response.content)
        (output / "result.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n")
        return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:7860")
    parser.add_argument("--mode", choices=("auto", "tab", "notation", "both"), default="auto")
    parser.add_argument("--session", help="Continue an existing native project")
    parser.add_argument("--action", choices=("process", "detect"), default="process")
    parser.add_argument('--manifest', type=Path,
                        help='Measure images and layout; pair with --model or saved --observations')
    parser.add_argument('--observations', type=Path, help='Assemble previously predicted observations')
    parser.add_argument('--model', type=Path, help='Merged observation OCR model')
    parser.add_argument('--auxiliary', type=Path, help='Explicit predicted state/voice mappings, keyed by id')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--limit', type=int, default=0, help='Soft measure limit; whole documents are retained')
    args = parser.parse_args()
    if args.manifest or args.observations:
        if args.inputs or args.session:
            parser.error('Research input cannot be combined with native files or session')
        if args.manifest and args.observations is None and args.model is None:
            parser.error('--manifest requires --model or --observations')
        rows = select_rows(read_rows(args.manifest or args.observations), args.limit)
        if not rows:
            parser.error('No observations to process')
        args.output.mkdir(parents=True, exist_ok=False)
        predictions = []
        if args.observations:
            saved = {row['id']: row for row in read_rows(args.observations)}
            stream = (observation_record(row, saved.get(row['id'], {}).get('raw',
                          saved.get(row['id'], {}).get('observation')),
                          saved.get(row['id'], {}).get('tokens')) for row in rows)
        else:
            stream = predict_observations(rows, args.model, args.device, args.batch_size, args.max_tokens)
        with (args.output / 'predictions.jsonl').open('w') as handle:
            for prediction in stream:
                predictions.append(prediction)
                handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                handle.flush()
        auxiliary = read_auxiliary(args.auxiliary, predictions) if args.auxiliary else None
        result = assemble_predictions(predictions, auxiliary)
        result['scope'] = 'research_observation_assembly'
        result['auxiliary'] = 'predicted' if auxiliary is not None else 'none'
        (args.output / 'assembled.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'records': len(predictions), 'auxiliary': result['auxiliary'],
                          'invalid_observations': sum(row.get('observation') is None for row in predictions)}))
        return
    if bool(args.inputs) == bool(args.session):
        parser.error("Supply input files or --session")
    with NativeClient(args.endpoint) as client:
        project = client.recognize(args.inputs, args.output, mode=args.mode, session=args.session,
                                   action=args.action)
    print(json.dumps({key: project.get(key) for key in (
        "id", "mode", "review_measures", "seconds",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
