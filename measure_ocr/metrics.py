from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from shared.m2 import (
    _duration_token,
    format_measure_target,
    parse_measure_target,
)
from shared.constraints import validate_measure_target


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _counter_overlap(left: Counter[Any], right: Counter[Any]) -> int:
    return sum((left & right).values())


def _f1(true_positive: int, predicted: int, expected: int) -> dict[str, float]:
    precision = _safe_div(true_positive, predicted)
    recall = _safe_div(true_positive, expected)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def _event_map(measure: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(voice["voice"]), int(event["start"])): event
        for voice in measure.get("voices", [])
        for event in voice.get("events", [])
    }


def _note_signature(note: dict[str, Any]) -> tuple[Any, ...]:
    return (
        note.get("string"),
        note.get("fret"),
        note.get("pitch"),
        tuple(sorted(note.get("effects") or [])),
        int(note.get("velocity", 95)),
        bool(note.get("swap_accidentals")),
    )


def _core_note_signature(note: dict[str, Any]) -> tuple[Any, ...]:
    return (note.get("string"), note.get("fret"), note.get("pitch"))


def _techniques(measure: dict[str, Any]) -> Counter[tuple[Any, ...]]:
    values: Counter[tuple[Any, ...]] = Counter()
    for voice in measure.get("voices", []):
        voice_index = int(voice["voice"])
        for event in voice.get("events", []):
            start = int(event["start"])
            for effect in event.get("effects") or []:
                values[(voice_index, start, "beat", effect)] += 1
            for note in event.get("notes", []):
                anchor = (note.get("string"), note.get("fret"), note.get("pitch"))
                for effect in note.get("effects") or []:
                    values[(voice_index, start, anchor, effect)] += 1
    return values


def _metadata(measure: dict[str, Any]) -> Counter[tuple[str, str]]:
    names = (
        "time_signature", "tempo_quarter", "key_signature", "triplet_feel",
        "bars", "repeat_count", "alternate_endings", "section", "direction",
        "from_direction",
    )
    return Counter(
        (name, repr(measure.get(name)))
        for name in names
        if measure.get(name) not in (None, [], "")
    )


def _technique_class(effect: str) -> str:
    return effect.split(":", 1)[0]


@dataclass
class MeasureSequenceMetrics:
    samples: int = 0
    syntax_valid: int = 0
    exact: int = 0
    core_exact: int = 0
    rhythm_exact: int = 0
    note_fields_exact: int = 0
    technique_exact: int = 0
    constraint_valid: int = 0
    metadata_fields: int = 0
    metadata_correct: int = 0
    metadata_expected: int = 0
    metadata_predicted: int = 0
    metadata_overlap: int = 0
    metadata_exact: int = 0
    event_expected: int = 0
    event_predicted: int = 0
    event_overlap: int = 0
    aligned_events: int = 0
    duration_correct: int = 0
    status_correct: int = 0
    note_expected: int = 0
    note_predicted: int = 0
    note_overlap: int = 0
    string_expected: int = 0
    string_correct: int = 0
    fret_expected: int = 0
    fret_correct: int = 0
    pitch_expected: int = 0
    pitch_correct: int = 0
    technique_expected: int = 0
    technique_predicted: int = 0
    technique_overlap: int = 0
    technique_class_expected: Counter[str] = field(default_factory=Counter)
    technique_class_predicted: Counter[str] = field(default_factory=Counter)
    technique_class_overlap: Counter[str] = field(default_factory=Counter)
    parse_errors: Counter[str] = field(default_factory=Counter)
    constraint_errors: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        expected_text: str,
        predicted_text: str,
        mode: str | None = None,
        *,
        tuning: list[int] | None = None,
        string_count: int | None = None,
    ) -> None:
        self.samples += 1
        expected_text = expected_text.strip()
        predicted_text = predicted_text.strip()
        try:
            expected = parse_measure_target(expected_text)
        except Exception as error:  # dataset construction failure
            raise ValueError(f"Invalid expected M2 target: {expected_text!r}") from error
        expected_metadata = _metadata(expected)
        self.metadata_expected += sum(expected_metadata.values())
        try:
            predicted = parse_measure_target(predicted_text)
        except Exception as error:
            self.parse_errors[error.__class__.__name__] += 1
            self._count_expected_only(expected)
            return
        self.syntax_valid += 1
        if mode is not None:
            _validated, constraint_errors = validate_measure_target(
                predicted_text,
                mode,
                tuning=tuning,
                string_count=string_count,
            )
            if not constraint_errors:
                self.constraint_valid += 1
            for error in constraint_errors:
                self.constraint_errors[error.split(":", 1)[0]] += 1
        if mode is not None:
            if format_measure_target(expected, mode) == format_measure_target(
                predicted, mode
            ):
                self.exact += 1
        elif predicted_text == expected_text:
            self.exact += 1

        predicted_metadata = _metadata(predicted)
        metadata_overlap = expected_metadata & predicted_metadata
        self.metadata_predicted += sum(predicted_metadata.values())
        self.metadata_overlap += sum(metadata_overlap.values())
        if expected_metadata == predicted_metadata:
            self.metadata_exact += 1

        metadata_names = (
            "time_signature", "tempo_quarter", "key_signature", "triplet_feel",
            "bars", "repeat_count", "alternate_endings", "section", "direction",
            "from_direction",
        )
        for name in metadata_names:
            expected_value = expected.get(name)
            if expected_value in (None, [], ""):
                continue
            self.metadata_fields += 1
            if predicted.get(name) == expected_value:
                self.metadata_correct += 1

        expected_events = _event_map(expected)
        predicted_events = _event_map(predicted)
        expected_keys = Counter(expected_events.keys())
        predicted_keys = Counter(predicted_events.keys())
        overlap_keys = set(expected_events) & set(predicted_events)
        same_event_keys = set(expected_events) == set(predicted_events)
        rhythm_exact = same_event_keys and all(
            _duration_token(expected_events[key]["duration"])
            == _duration_token(predicted_events[key]["duration"])
            and expected_events[key].get("status")
            == predicted_events[key].get("status")
            for key in expected_events
        )
        note_fields_exact = same_event_keys and all(
            Counter(
                _core_note_signature(note)
                for note in expected_events[key].get("notes", [])
            )
            == Counter(
                _core_note_signature(note)
                for note in predicted_events[key].get("notes", [])
            )
            for key in expected_events
        )
        if rhythm_exact:
            self.rhythm_exact += 1
        if note_fields_exact:
            self.note_fields_exact += 1
        if expected_metadata == predicted_metadata and rhythm_exact and note_fields_exact:
            self.core_exact += 1
        self.event_expected += len(expected_events)
        self.event_predicted += len(predicted_events)
        self.event_overlap += _counter_overlap(expected_keys, predicted_keys)
        self.aligned_events += len(overlap_keys)
        for key in overlap_keys:
            expected_event = expected_events[key]
            predicted_event = predicted_events[key]
            if _duration_token(expected_event["duration"]) == _duration_token(predicted_event["duration"]):
                self.duration_correct += 1
            if expected_event.get("status") == predicted_event.get("status"):
                self.status_correct += 1
            self._compare_notes(expected_event, predicted_event)
        for key in set(expected_events) - overlap_keys:
            self._compare_notes(expected_events[key], {"notes": []})
        for key in set(predicted_events) - overlap_keys:
            self.note_predicted += len(predicted_events[key].get("notes", []))

        expected_techniques = _techniques(expected)
        predicted_techniques = _techniques(predicted)
        overlapping_techniques = expected_techniques & predicted_techniques
        if expected_techniques == predicted_techniques:
            self.technique_exact += 1
        self.technique_expected += sum(expected_techniques.values())
        self.technique_predicted += sum(predicted_techniques.values())
        self.technique_overlap += sum(overlapping_techniques.values())
        for key, count in expected_techniques.items():
            self.technique_class_expected[_technique_class(str(key[-1]))] += count
        for key, count in predicted_techniques.items():
            self.technique_class_predicted[_technique_class(str(key[-1]))] += count
        for key, count in overlapping_techniques.items():
            self.technique_class_overlap[_technique_class(str(key[-1]))] += count

    def _count_expected_only(self, expected: dict[str, Any]) -> None:
        expected_events = _event_map(expected)
        self.event_expected += len(expected_events)
        for event in expected_events.values():
            self._compare_notes(event, {"notes": []})
        techniques = _techniques(expected)
        self.technique_expected += sum(techniques.values())
        for key, count in techniques.items():
            self.technique_class_expected[_technique_class(str(key[-1]))] += count

    def _compare_notes(self, expected_event: dict[str, Any], predicted_event: dict[str, Any]) -> None:
        expected_notes = list(expected_event.get("notes", []))
        predicted_notes = list(predicted_event.get("notes", []))
        self.note_expected += len(expected_notes)
        self.note_predicted += len(predicted_notes)
        expected_counter = Counter(_note_signature(note) for note in expected_notes)
        predicted_counter = Counter(_note_signature(note) for note in predicted_notes)
        self.note_overlap += _counter_overlap(expected_counter, predicted_counter)
        for name in ("string", "fret", "pitch"):
            expected_with_field = [note for note in expected_notes if name in note]
            predicted_with_field = [note for note in predicted_notes if name in note]
            if name == "string":
                expected_values = Counter(note["string"] for note in expected_with_field)
                predicted_values = Counter(note["string"] for note in predicted_with_field)
            elif name == "fret":
                expected_values = Counter(
                    (note.get("string"), note["fret"]) for note in expected_with_field
                )
                predicted_values = Counter(
                    (note.get("string"), note["fret"]) for note in predicted_with_field
                )
            else:
                expected_values = Counter(
                    (note.get("string"), note["pitch"])
                    if "string" in note else note["pitch"]
                    for note in expected_with_field
                )
                predicted_values = Counter(
                    (note.get("string"), note["pitch"])
                    if "string" in note else note["pitch"]
                    for note in predicted_with_field
                )
            setattr(
                self, f"{name}_expected",
                getattr(self, f"{name}_expected") + sum(expected_values.values()),
            )
            setattr(
                self, f"{name}_correct",
                getattr(self, f"{name}_correct") + _counter_overlap(
                    expected_values, predicted_values
                ),
            )

    def result(self) -> dict[str, Any]:
        technique_classes = sorted(
            set(self.technique_class_expected) | set(self.technique_class_predicted)
        )
        return {
            "samples": self.samples,
            "syntax_valid_rate": _safe_div(self.syntax_valid, self.samples),
            "exact_match_rate": _safe_div(self.exact, self.samples),
            "core_exact_rate": _safe_div(self.core_exact, self.samples),
            "rhythm_exact_rate": _safe_div(self.rhythm_exact, self.samples),
            "note_fields_exact_rate": _safe_div(self.note_fields_exact, self.samples),
            "technique_exact_rate": _safe_div(self.technique_exact, self.samples),
            "constraint_valid_rate": _safe_div(self.constraint_valid, self.samples),
            "metadata_accuracy": _safe_div(self.metadata_correct, self.metadata_fields),
            "metadata_exact_rate": _safe_div(self.metadata_exact, self.samples),
            "metadata_field": _f1(
                self.metadata_overlap, self.metadata_predicted, self.metadata_expected
            ),
            "event_onset": _f1(self.event_overlap, self.event_predicted, self.event_expected),
            "duration_accuracy_on_aligned_events": _safe_div(self.duration_correct, self.aligned_events),
            "status_accuracy_on_aligned_events": _safe_div(self.status_correct, self.aligned_events),
            "note_exact": _f1(self.note_overlap, self.note_predicted, self.note_expected),
            "string_accuracy": _safe_div(self.string_correct, self.string_expected),
            "fret_accuracy": _safe_div(self.fret_correct, self.fret_expected),
            "pitch_accuracy": _safe_div(self.pitch_correct, self.pitch_expected),
            "technique": _f1(
                self.technique_overlap, self.technique_predicted, self.technique_expected
            ),
            "technique_by_class": {
                name: {
                    **_f1(
                        self.technique_class_overlap[name],
                        self.technique_class_predicted[name],
                        self.technique_class_expected[name],
                    ),
                    "expected": self.technique_class_expected[name],
                    "predicted": self.technique_class_predicted[name],
                }
                for name in technique_classes
            },
            "counts": {
                "expected_events": self.event_expected,
                "predicted_events": self.event_predicted,
                "expected_notes": self.note_expected,
                "predicted_notes": self.note_predicted,
            },
            "parse_errors": dict(self.parse_errors),
            "constraint_errors": dict(self.constraint_errors),
        }
