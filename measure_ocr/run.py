"""Recognize ordered crops and write score text plus the saved model sequence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from shared.defaults import MODEL, MEASURE_ADAPTER

from measure_ocr.recognizer import recognize_crops
from shared.artifacts import read_result, write_json, write_result
from shared.m2 import format_measure_target, parse_measure_target
from shared.score_text import display_score_text


def run(
    layout: Path,
    info: Path,
    output: Path,
    *,
    model: Path = MODEL,
    adapter: Path | None = MEASURE_ADAPTER,
    device: str = "cuda",
    max_new_tokens: int = 512,
    max_new_tokens_ceiling: int = 2048,
    maximum_attempts: int = 3,
    resume: bool = False,
    backend=None,
    progress=None,
    initial_records=None,
    retry_measures=None,
    cancelled=None,
) -> Path:
    if (
        max_new_tokens <= 0
        or max_new_tokens_ceiling < max_new_tokens
        or maximum_attempts < 1
    ):
        raise ValueError(
            "Token limits and attempt count must be positive; ceiling must be >= initial limit"
        )
    source = read_result(layout, "layout")
    information = read_result(info, "document_info")
    if Path(information["layout"]).resolve() != layout.resolve():
        raise ValueError("Document information belongs to a different layout result")
    records = source["records"]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "recognition.jsonl"
    # Reusing accepted measures requires the same crops, metadata, models and options.
    context = sha256(layout.read_bytes() + info.read_bytes())
    for row in records:
        context.update(Path(row["image"]).read_bytes())
    for path in (model, adapter):
        if path is not None:
            context.update(str(path.resolve()).encode())
            for name in (
                "config.json",
                "adapter_config.json",
                "adapter_model.safetensors",
                "model.safetensors",
            ):
                artifact = path / name
                if artifact.is_file():
                    stat = artifact.stat()
                    context.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    context.update(
        f"{device}:{max_new_tokens}:{max_new_tokens_ceiling}:{maximum_attempts}".encode()
    )
    context.update(
        json.dumps(
            {"initial": initial_records, "retry": retry_measures}, sort_keys=True
        ).encode()
    )
    signature = context.hexdigest()
    signature_path = output / "recognition_context.json"
    if resume and log.is_file():
        previous = (
            json.loads(signature_path.read_text(encoding="utf-8"))
            if signature_path.is_file()
            else {}
        )
        if previous.get("signature") != signature:
            raise ValueError(
                "Recognition inputs or options changed; use a new output or omit --resume"
            )
    if not resume:
        log.unlink(missing_ok=True)
    write_json(signature_path, {"signature": signature})
    targets = recognize_crops(
        records,
        source["mode"],
        model.resolve(),
        adapter.resolve() if adapter else None,
        device,
        max_new_tokens,
        max_new_tokens_ceiling,
        information["tuning_used"],
        maximum_attempts,
        log,
        resume,
        backend,
        progress,
        initial_records,
        retry_measures,
        cancelled,
    )
    metadata = information["document_metadata"]
    if targets and metadata.get("tempo_quarter"):
        first = parse_measure_target(targets[0])
        first["tempo_quarter"] = int(metadata["tempo_quarter"])
        targets[0] = format_measure_target(
            first, records[0].get("mode") or source["mode"], preserve_playback=True
        )
    for row, target in zip(records, targets):
        row["target"] = target
    review = [row["measure_number"] for row in records if row.get("needs_review")]
    m2_path = output / "prediction.m2"
    m2_path.write_text("\n".join(targets) + "\n", encoding="utf-8")
    score_path = output / "score.txt"
    score_path.write_text(display_score_text("\n".join(targets) + "\n"), encoding="utf-8")
    return write_result(
        output,
        "measure_ocr",
        layout=str(layout.resolve()),
        info=str(info.resolve()),
        mode=source["mode"],
        measures=len(records),
        m2=str(m2_path),
        score_text=str(score_path),
        recognition_log=str(log),
        records=records,
        review_measures=review,
        status="needs_review" if review else "complete",
        **{
            key: information[key]
            for key in ("document_metadata", "title", "artist", "tuning_used", "capo")
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize crops from layout and document-info manifests."
    )
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=MEASURE_ADAPTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens-ceiling", type=int, default=2048)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
