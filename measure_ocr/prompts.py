PROMPTS = {
    "tab": (
        "Guitar TAB measure recognition: return exactly one M2 fragment. Preserve every voice, "
        "event start, duration, string, fret, rest, tie, and visible technique."
    ),
    "both": (
        "Guitar score+TAB measure recognition: return exactly one M2 fragment. Preserve every "
        "voice, event start, duration, string, fret, pitch, rest, tie, and visible technique."
    ),
    "notation": (
        "Guitar notation measure recognition: return exactly one M2 fragment. Preserve every "
        "voice, event start, duration, pitch, rest, tie, and visible technique."
    ),
}


def recognition_prompt(mode: str, previous_context: str | None = None) -> str:
    prompt = PROMPTS[mode]
    if previous_context is not None:
        prompt += f" Previous measure context: {previous_context}"
    return prompt
