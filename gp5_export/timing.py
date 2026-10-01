"""GP5 timing diagnostics; these are not restrictions on legal M2 scores."""

from shared.m2 import duration_ticks, parse_measure_target


def gp5_timing_errors(target: str) -> list[str]:
    """Find events that GP5's sequential beat storage cannot preserve."""
    measure = parse_measure_target(target)
    errors = []
    for voice in measure["voices"]:
        end = 0
        for index, event in enumerate(voice["events"]):
            if event["start"] < end:
                errors.append(f"声部 {voice['voice'] + 1} 的第 {index + 1} 个事件与前一个重叠")
            end = event["start"] + int(duration_ticks(event["duration"]))
    return errors
