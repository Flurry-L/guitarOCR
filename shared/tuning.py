"""Guitar tuning definitions shared by recognition and export."""

DEFAULT_TUNING = (64, 59, 55, 50, 45, 40)


TUNING_NAMES = {
    "standard": list(DEFAULT_TUNING),
    "standard tuning": list(DEFAULT_TUNING),
    "dropped d": [64, 59, 55, 50, 45, 38],
    "drop d": [64, 59, 55, 50, 45, 38],
    "tune down 1/2 step": [63, 58, 54, 49, 44, 39],
    "tune down 1 step": [62, 57, 53, 48, 43, 38],
    "tune down 2 step": [60, 55, 51, 46, 41, 36],
}


def tuning_from_name(name: str | None) -> list[int] | None:
    if not name:
        return None
    return TUNING_NAMES.get(name.strip().casefold())
