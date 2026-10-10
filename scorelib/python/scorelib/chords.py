"""Visible chord symbols and diagrams; string arrays run low to high."""

import json
import re
from urllib.parse import unquote

from scorelib import _native


def normalize_diagram(value):
    if not isinstance(value, dict):
        return None
    return json.loads(_native.normalize_diagram(json.dumps(value)))


def diagram_effect(diagram):
    return _native.diagram_effect(json.dumps(diagram))


def parse_diagram_effect(effect):
    return json.loads(_native.parse_diagram_effect(effect))


def chord_key(name):
    return re.sub(r"\s+", "", str(name or "")).replace("♭", "b").replace("♯", "#")


def event_chord(event):
    effects = event.get("effects", [])
    name = next((unquote(e[6:]) for e in effects if e.startswith("chord:")), None)
    diagram = next(
        (parse_diagram_effect(e) for e in effects if e.startswith("diagram:")), None
    )
    return name, diagram
