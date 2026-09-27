"""Named tunings shared by recognition and export."""

DEFAULT_TUNING = (64, 59, 55, 50, 45, 40)


TUNING_NAMES = {
    "standard": list(DEFAULT_TUNING),
    "standard tuning": list(DEFAULT_TUNING),
    "dropped d": [64, 59, 55, 50, 45, 38],
    "drop d": [64, 59, 55, 50, 45, 38],
    "dropped c": [62, 57, 53, 48, 43, 36],
    "drop c": [62, 57, 53, 48, 43, 36],
    "dropped b": [61, 56, 52, 47, 42, 35],
    "drop b": [61, 56, 52, 47, 42, 35],
    "dropped d tune down 1/2 step": [63, 58, 54, 49, 44, 37],
    "tune down 1/2 step": [63, 58, 54, 49, 44, 39],
    "tune down 1 step": [62, 57, 53, 48, 43, 38],
    "tune down 2 step": [60, 55, 51, 46, 41, 36],
}


def tuning_from_name(
    name: str | None, instrument: str = "guitar", string_count: int | None = None
) -> list[int] | None:
    if not name:
        return None
    label = name.strip().casefold()
    if instrument == "guitar" and string_count in {None, 6}:
        return TUNING_NAMES.get(label)
    from shared.instruments import standard_tuning

    if instrument not in {"guitar", "bass"}:
        return None
    try:
        tuning = standard_tuning(instrument, string_count)
    except ValueError:
        return None
    if label in {"standard", "standard tuning"}:
        return tuning
    shifts = {"tune down 1/2 step": -1, "tune down 1 step": -2, "tune down 2 step": -4}
    if label in shifts:
        return [pitch + shifts[label] for pitch in tuning]
    if instrument == "bass" and len(tuning) == 4:
        drop_shifts = {
            "drop d": 0,
            "dropped d": 0,
            "drop c": -2,
            "dropped c": -2,
            "drop b": -3,
            "dropped b": -3,
            "dropped d tune down 1/2 step": -1,
        }
        if label in drop_shifts:
            return [pitch + drop_shifts[label] for pitch in [*tuning[:-1], 26]]
    return None
