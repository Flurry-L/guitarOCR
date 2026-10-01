"""Soft review hints for accepted OCR results, separate from M2 validity."""

from fractions import Fraction

from shared.m2 import duration_ticks, parse_measure_target


def ocr_rhythm_warnings(target: str, time_signature: str = "4/4") -> list[str]:
    """Flag overfull voices for review without rejecting historical M2 or pickups."""
    measure = parse_measure_target(target)
    signature = measure.get("time_signature") or time_signature
    numerator, denominator = map(int, signature.split("/"))
    limit = Fraction(3840 * numerator, denominator)
    warnings = []
    for voice in measure["voices"]:
        end = max(
            (event["start"] + duration_ticks(event["duration"]) for event in voice["events"]),
            default=Fraction(0),
        )
        if end > limit:
            warnings.append(f"声部 {voice['voice'] + 1} 的音符超出 {signature} 小节时长，请核对起点与时值")
    return warnings
