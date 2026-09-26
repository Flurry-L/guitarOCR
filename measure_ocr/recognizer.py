from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from measure_ocr.prompts import recognition_prompt
from shared.m2 import format_history_context, parse_measure_target, full_measure_rest_target as _full_measure_rest_target
from shared.constraints import validate_measure_target
from shared.glm_backend import GlmBackend


def _repair_truncated_optional_text(target: str) -> tuple[str, list[str]]:
    """Drop only an unterminated optional text effect and close open voices."""

    text_start = target.rfind("<text:")
    if text_start < 0 or target.find(">", text_start) >= 0:
        return target, []
    repaired = target[:text_start].rstrip()
    missing_voice_closers = max(0, repaired.count("{") - repaired.count("}"))
    repaired += "}" * missing_voice_closers
    return repaired, ["drop_unterminated_text"]


def _active_time_signature(previous_targets: list[str]) -> tuple[int, int]:
    for target in reversed(previous_targets):
        value = parse_measure_target(target).get("time_signature")
        if value:
            numerator, denominator = str(value).split("/", maxsplit=1)
            return max(1, int(numerator)), max(1, int(denominator))
    return 4, 4


def _rest_fallback_target(previous_targets: list[str]) -> str:
    """Keep swing through a failed bar without inheriting it over valid straight bars.

    M2 feel describes each bar: omission in a valid prediction means straight.
    A failed prediction provides no such evidence. Its reviewed rest placeholder
    provisionally keeps only the immediately preceding bar's feel.
    """
    target = _full_measure_rest_target(_active_time_signature(previous_targets))
    feel = parse_measure_target(previous_targets[-1]).get("triplet_feel") if previous_targets else None
    if feel:
        target = target.replace("M2", "M2 feel=" + str(feel), 1)
    return target


def recognize_crops(
    records: list[dict[str, Any]],
    mode: str,
    model_path: Path,
    adapter_path: Path | None,
    device: str,
    max_new_tokens: int,
    max_new_tokens_ceiling: int,
    tuning: list[int],
    maximum_attempts: int,
    diagnostics_path: Path,
    resume: bool,
    backend=None,
    progress=None,
    initial_records=None,
    retry_measures=None,
    cancelled=None,
) -> list[str]:
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    if diagnostics_path.is_file() and not resume:
        diagnostics_path.unlink()
    accepted: dict[int, dict[str, Any]] = {}
    if diagnostics_path.is_file():
        lines = diagnostics_path.read_bytes().splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines):
            if not line.strip():
                offset += len(line)
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError(
                        "Recognition log is damaged before its final partial record"
                    ) from None
                with diagnostics_path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            if value.get("accepted"):
                accepted[int(value["measure_number"])] = value
            offset += len(line)
        # An otherwise complete JSON record without a newline must be separated before appending.
        if (
            diagnostics_path.stat().st_size
            and not diagnostics_path.read_bytes().endswith(b"\n")
        ):
            with diagnostics_path.open("ab") as handle:
                handle.write(b"\n")

    retry = set(retry_measures or [])
    seeds = {int(row["measure_number"]): row for row in (initial_records or [])}
    if not retry.issubset({int(row["measure_number"]) for row in records}):
        raise ValueError("Invalid measure number to retry")
    targets: list[str] = []
    default_mode = mode
    with diagnostics_path.open("a", encoding="utf-8") as diagnostics:
        for index, record in enumerate(records, start=1):
            mode = record.get("mode") or default_mode
            if mode not in {"tab", "notation", "both"}:
                raise ValueError(f"Measure {record['measure_number']} has no notation type")
            record["mode"] = mode
            # Context note fields differ between notation modes. Restart the
            # visual context at a mode change instead of inventing fingerings
            # or trying to format a pitch-only note as TAB.
            previous_context = "START"
            if targets and (records[index - 2].get("mode") or default_mode) == mode:
                previous_context = format_history_context(targets, mode)
            if cancelled and cancelled():
                from shared.tasks import Cancelled

                raise Cancelled("识别已停止，已完成的小节已保存，可以继续")
            number = int(record["measure_number"])
            saved = accepted.get(number)
            seed = seeds.get(number) if number not in retry else None
            if saved or seed:
                value = saved or seed
                target = str(value["target"])
                _, errors = validate_measure_target(
                    target, mode, tuning=tuning, string_count=len(tuning)
                )
                if errors:
                    raise ValueError(f"Invalid saved measure {number}: {errors}")
                record.update(
                    {
                        key: value[key]
                        for key in (
                            "manually_edited",
                            "needs_review",
                            "fallback_reason",
                        )
                        if key in value
                    }
                )
                if saved:
                    record["needs_review"] = bool(value.get("fallback_reason"))
                record["target"] = target
                record["previous_context"] = previous_context
                record["recognition_attempts"] = value.get(
                    "attempt", value.get("recognition_attempts", 0)
                )
                targets.append(target)
                if progress:
                    progress(index, len(records))
                continue
            backend = backend or GlmBackend(model_path, adapter_path, device)
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": record["image"]},
                        {
                            "type": "text",
                            "text": recognition_prompt(mode, previous_context),
                        },
                    ],
                }
            ]
            target = ""
            constraint_errors: list[str] = []
            token_budget = max_new_tokens
            for attempt in range(1, maximum_attempts + 1):
                if cancelled and cancelled():
                    from shared.tasks import Cancelled

                    raise Cancelled("识别已停止，已完成的小节已保存，可以继续")
                value, generated_token_count = backend.generate(messages, token_budget)
                hit_token_limit = generated_token_count >= token_budget
                value = value.strip()
                start = value.find("M2")
                target = value[start:].strip() if start >= 0 else value
                target, deterministic_repairs = _repair_truncated_optional_text(target)
                _parsed, constraint_errors = validate_measure_target(
                    target,
                    mode,
                    tuning=tuning,
                    string_count=len(tuning),
                )
                if (
                    not constraint_errors
                    and deterministic_repairs
                    and hit_token_limit
                    and attempt < maximum_attempts
                ):
                    constraint_errors = [
                        "generation reached max_new_tokens inside optional text"
                    ]
                accepted_attempt = not constraint_errors
                diagnostics.write(
                    json.dumps(
                        {
                            "measure_number": int(record["measure_number"]),
                            "mode": mode,
                            "image": record["image"],
                            "attempt": attempt,
                            "token_budget": token_budget,
                            "generated_token_count": generated_token_count,
                            "hit_token_limit": hit_token_limit,
                            "raw": value,
                            "target": target,
                            "deterministic_repairs": deterministic_repairs,
                            "constraint_errors": constraint_errors,
                            "accepted": accepted_attempt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                diagnostics.flush()
                if accepted_attempt:
                    break
                if attempt < maximum_attempts:
                    if constraint_errors == [
                        "generation reached max_new_tokens inside optional text"
                    ]:
                        correction = (
                            "The optional text annotation exhausted the output limit. Ignore all "
                            "text, section, and chord-name annotations. Return the same musical "
                            "events as one complete M2 fragment without any text effect."
                        )
                    else:
                        correction = (
                            "Correct the M2 using the same image. Constraint errors: "
                            + "; ".join(constraint_errors[:8])
                            + ". Return only one corrected M2 fragment."
                        )
                        if hit_token_limit:
                            token_budget = min(max_new_tokens_ceiling, token_budget * 2)
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": target}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": correction,
                                    }
                                ],
                            },
                        ]
                    )
            if constraint_errors:
                target = _rest_fallback_target(targets)
                _parsed, fallback_errors = validate_measure_target(
                    target,
                    mode,
                    tuning=tuning,
                    string_count=len(tuning),
                )
                if fallback_errors:
                    raise ValueError(
                        f"Measure {record['measure_number']} failed M2 constraints after "
                        f"{maximum_attempts} attempts and its rest fallback was invalid: "
                        f"{fallback_errors}"
                    )
                diagnostics.write(
                    json.dumps(
                        {
                            "measure_number": int(record["measure_number"]),
                            "mode": mode,
                            "image": record["image"],
                            "attempt": maximum_attempts + 1,
                            "token_budget": 0,
                            "generated_token_count": 0,
                            "hit_token_limit": False,
                            "raw": "",
                            "target": target,
                            "deterministic_repairs": ["fallback_full_measure_rest"],
                            "constraint_errors": [],
                            "fallback_reason": constraint_errors,
                            "accepted": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                diagnostics.flush()
            targets.append(target)
            record["target"] = target
            record["previous_context"] = previous_context
            record["recognition_attempts"] = attempt
            record["needs_review"] = bool(constraint_errors)
            record["fallback_reason"] = constraint_errors
            if progress:
                progress(index, len(records))
            print(
                f"[{index}/{len(records)}] measure {record['measure_number']}",
                flush=True,
            )
    return targets
