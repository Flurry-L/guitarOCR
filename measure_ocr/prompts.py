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


def recognition_prompt(mode: str, previous_context: str | None = None, instrument: str = "guitar", pitch_context: dict | None = None, *, written_pitch: bool = False) -> str:
    prompt = PROMPTS[mode]
    if instrument == "bass":
        prompt = prompt.replace("Guitar", "Bass")
    elif instrument in {"pitched", "drums"}:
        if mode != "notation":
            raise ValueError(f"{instrument} requires notation")
        prompt = prompt.replace("Guitar", "Pitched instrument" if instrument == "pitched" else "Drum")
        if instrument == "drums":
            prompt += " Pitch fields are General MIDI percussion keys, not melodic pitches."
    elif instrument != "guitar":
        raise ValueError(f"Unsupported instrument: {instrument}")
    if pitch_context is not None and mode != "tab" and instrument != "drums":
        import json
        from shared.pitch_context import prompt_pitch_context

        pitch_instruction = (
            ". Return written MIDI pitches from staff positions and the printed key signature. "
            "Do not apply instrument transposition, clef octave, or ottava to these pitches. "
            "Use the written key signature. Previous measure context also uses written pitches. "
            if written_pitch and mode == "notation" else
            ". Return sounding MIDI pitches and concert key signatures. "
            "Instrument transposition, clef octave, and local ottava each apply once. "
        )
        prompt += (
            " Pitch context: " + json.dumps(prompt_pitch_context(pitch_context), separators=(",", ":"))
            + pitch_instruction +
            "Preserve local octave markings as ottava:12, ottava:-12, ottava:24, or ottava:-24 "
            "on every affected event. Span start/end are fractions of the measure width."
        )
    if previous_context is not None:
        if written_pitch and pitch_context is not None and mode == "notation" and instrument != "drums" and previous_context != "START":
            from shared.pitch_context import convert_pitch_target

            previous_context = convert_pitch_target(previous_context.replace("C2", "M2", 1), pitch_context, to_written=True).replace("M2", "C2", 1)
        prompt += f" Previous measure context: {previous_context}"
    return prompt
