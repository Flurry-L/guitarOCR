"""User-facing explanations for music validation; decoder error codes stay stable."""

import re


_CONSTRAINTS = {
    "unsupported_mode": "请选择 TAB、五线谱或混合谱",
    "invalid_time_signature": "拍号格式有误，例如 4/4 或 6/8",
    "invalid_tempo": "小节速度须为 1 至 999 的整数",
    "missing_voice": "请至少保留一个声部",
    "unsupported_gp5_voice": "GP5 最多支持两个声部（V0、V1）",
    "duplicate_voice": "同一声部重复出现，请合并后保存",
    "empty_voice": "声部中至少需要一个音符或休止",
    "negative_event_start": "音符或休止的起点不能小于 0",
    "non_increasing_event_starts": "同一声部的起点须递增，同时发声的音符请写在同一行",
    "invalid_duration": "请检查时值和连音比例",
    "unknown_beat_effect": "拍级奏法格式有误，请参照小节文本格式说明",
    "unknown_note_effect": "音符奏法格式有误，请参照小节文本格式说明",
    "tab_note_fields": "TAB 音符需要弦号和品位",
    "notation_note_fields": "五线谱音符请填写 MIDI 音高",
    "both_note_fields": "混合谱音符需要弦号、品位和 MIDI 音高",
    "invalid_string": "弦号超出当前调弦的弦数",
    "invalid_fret": "品位须为 0 至 36，闷音用 x 表示",
    "pitch_string_fret_conflict": "音高与弦号、品位不一致，请检查音符或调弦",
    "invalid_pitch": "MIDI 音高须为 0 至 127",
    "invalid_velocity": "力度须为 0 至 127",
    "duplicate_string_in_chord": "同一和弦不能在同一根弦上放两个音符",
}


def _syntax_message(error: str) -> str:
    if "target must start with" in error:
        return "小节须以 MEASURE 开头，并用 | 分隔小节头和声部"
    if "repeated payload" in error:
        return "声部中的第一项需要写出音符或休止，不能使用重复标记 ^"
    for prefix, label in {
        "Invalid M2 duration token:": "时值格式有误",
        "Invalid M2 note token:": "音符格式有误",
        "Invalid M2 metadata token:": "小节头字段格式有误",
        "Unknown M2 metadata:": "不支持的小节头字段",
        "Duplicate M2 metadata:": "重复的小节头字段",
        "Invalid M2 voice:": "声部格式有误",
        "Invalid M2 event:": "音符或休止格式有误",
        "Invalid M2 event start:": "起点格式有误",
        "Invalid M2 event duration:": "时值格式有误",
    }.items():
        if error.startswith(prefix):
            return label + "：" + error.removeprefix(prefix).strip()
    return "小节文本格式有误，请检查小节头、声部和音符"


def correction_message(errors: list[str]) -> str:
    messages = []
    for error in errors:
        code = error.split(":", 1)[0]
        message = (
            _syntax_message(error.partition(":")[2]) if code == "syntax"
            else _CONSTRAINTS.get(code, "小节文本格式有误，请检查小节头、声部和音符")
        )
        location = re.search(r":V(\d+)(?::E(\d+))?(?::N(\d+))?$", error)
        if location:
            parts = [f"声部 {int(location[1]) + 1}"]
            if location[2] is not None:
                parts.append(f"第 {int(location[2]) + 1} 项")
            if location[3] is not None:
                parts.append(f"第 {int(location[3]) + 1} 个音符")
            message = "，".join(parts) + "：" + message
        messages.append(message + "。")
    return "\n".join(dict.fromkeys(messages))
