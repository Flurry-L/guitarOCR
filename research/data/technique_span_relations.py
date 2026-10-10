"""Compile draw-time text/endpoint supervision; never used for inference labels."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from research.common.pdf import open_pdf
from PIL import Image, ImageDraw


def pdf_spans(page):
    anchors = []
    words = page.words()
    for index, word in enumerate(words):
        left, top, right, bottom, text = word
        text = text.strip().lower().replace(" ", "")
        if text == "let" and index + 1 < len(words):
            x0, y0, x1, y1, following = words[index + 1]
            if following.casefold() == "ring" and abs(top - y0) < 1 and 0 <= x0 - right <= 6:
                right, bottom, text = x1, max(bottom, y1), "letring"
        kind = "pm" if text in ("p.m.", "pm") else "let" if text == "letring" else None
        if kind:
            anchors.append({"kind": kind, "text_bbox": [left, top, right, bottom]})
    dashes = page.dashed_lines()
    spans = []
    for anchor in anchors:
        left, top, right, bottom = anchor["text_bbox"]
        matching = [
            line
            for line in dashes
            if 0 <= line[0] - right <= 6 and top <= line[1] <= bottom
        ]
        if len(matching) == 1:
            spans.append(
                {
                    **anchor,
                    "start_x": (left + right) / 2,
                    "end_x": matching[0][2],
                    "line_y": matching[0][1],
                    "style": "labelled_dash",
                }
            )
        elif not matching:
            # A visible word supplies technique identity but no scope endpoint.
            # The annotation compiler keeps its downstream effect unknown.
            spans.append({**anchor, 'start_x': (left+right)/2,
                          'end_x': right, 'line_y': (top+bottom)/2,
                          'style': 'label_only'})
    return spans



def previous_image(row):
    path = Path(row["image"])
    if path.stem.startswith("system-"):
        index = int(path.stem.split("-")[-1])
        previous = path.with_name(f"system-{index - 1}.png")
        if previous.exists():
            return str(previous)
    return None


@lru_cache(maxsize=256)
def authored_spans(path):
    return json.loads(Path(path).read_text())["techniqueSpans"]


@lru_cache(maxsize=512)
def native_spans(path, page):
    with open_pdf(path) as doc:
        return pdf_spans(doc[page - 1])


def translated_bbox(box, crop):
    x, y, width, height = box
    return [x - crop[0], y - crop[1], x + width - crop[0], y + height - crop[1]]


def compile_row(row, previous_row=None):
    row = dict(row)
    x0, y0, x1, y1 = row["bbox"]
    spans = []
    previous_spans = []
    if not row.get("source_pdf"):
        path = Path(row["source_page"]).parent / "geometry.json"
        for raw in authored_spans(str(path)):
            box = raw["bbox"]
            if previous_row is not None:
                px0, py0, px1, py1 = previous_row["bbox"]
                if py0 <= box[1] + box[3] / 2 <= py1:
                    texts = [c["bbox"] for c in raw["commands"] if c["type"] == "text"]
                    if texts:
                        previous_spans.append(
                            dict(
                                kind=raw["kind"],
                                events=raw["events"],
                                anchor=translated_bbox(texts[0], previous_row["bbox"]),
                            )
                        )
            if not (
                y0 <= box[1] + box[3] / 2 <= y1 and x0 <= box[0] + box[2] / 2 <= x1
            ):
                continue
            texts = [c["bbox"] for c in raw["commands"] if c["type"] == "text"]
            if not texts:
                continue
            anchor = translated_bbox(texts[0], row["bbox"])
            extent = translated_bbox(box, row["bbox"])
            # The visible text center marks the printed start; the far edge is
            # supervised from actual painter commands, including terminal hooks.
            geometry = [
                (anchor[0] + anchor[2]) / 2,
                (extent[1] + extent[3]) / 2,
                extent[2],
                (extent[1] + extent[3]) / 2,
                *anchor,
            ]
            spans.append(
                dict(kind=raw["kind"], geometry=geometry, events=raw["events"])
            )
    else:
        for raw in native_spans(row["source_pdf"], row["page"]):
            factor = 180 / 72
            left, top, right, bottom = [v * factor for v in raw["text_bbox"]]
            line_y = raw["line_y"] * factor
            if not (y0 <= line_y <= y1):
                continue
            spans.append(
                dict(
                    kind=raw["kind"],
                    style=raw.get("style", "labelled_dash"),
                    geometry=[
                        raw["start_x"] * factor - x0,
                        line_y - y0,
                        raw["end_x"] * factor - x0,
                        line_y - y0,
                        left - x0,
                        top - y0,
                        right - x0,
                        bottom - y0,
                    ],
                    events=None,
                )
            )
    events = []
    for event in row["events"]:
        selected, present, continuations, styles = [], [], [], []
        for kind in ("pm", "let"):
            candidates = [s for s in spans if s["kind"] == kind]
            if candidates:

                def distance(span):
                    g = span["geometry"]
                    px, py = event["pixel"]
                    owned = any(
                        e["mi"] == event["measure_index"]
                        and e["vi"] == event["voice"]
                        and e["ei"] == event.get("event_index")
                        for e in span["events"] or []
                    )
                    horizontal = max(g[0] - px, px - g[2], 0)
                    return (not owned, horizontal, abs(py - g[1]))

                chosen = min(candidates, key=distance)
                selected.append(chosen["geometry"])
                present.append(1)
                styles.append(chosen.get("style", "renderer"))
                linked = [
                    s
                    for s in previous_spans
                    if s["kind"] == kind
                    and any(
                        e["mi"] == event["measure_index"]
                        and e["vi"] == event["voice"]
                        and e["ei"] == event.get("event_index")
                        for e in s["events"]
                    )
                ]
                if not linked and previous_row is not None and chosen["events"]:
                    # alphaTab closes its painter chain at a system break.
                    # Authoring-side adjacency can declare a continuous effect
                    # across that break; it never supplies inference labels.
                    current_voice = [
                        e for e in row["events"] if e["voice"] == event["voice"]
                    ]
                    previous_voice = [
                        e
                        for e in previous_row["events"]
                        if e["voice"] == event["voice"]
                    ]
                    if previous_voice:
                        first = min(
                            current_voice,
                            key=lambda e: (e["measure_index"], e["onset"]),
                        )
                        last = max(
                            previous_voice,
                            key=lambda e: (e["measure_index"], e["onset"]),
                        )
                        continuous = (
                            first["measure_index"] == last["measure_index"] + 1
                            and first["onset"] == 0
                            and first["label"] in (kind, "both")
                            and last["label"] in (kind, "both")
                            and any(
                                e["mi"] == first["measure_index"]
                                and e["vi"] == first["voice"]
                                and e["ei"] == first["event_index"]
                                for e in chosen["events"]
                            )
                        )
                        if continuous:
                            linked = [
                                s
                                for s in previous_spans
                                if s["kind"] == kind
                                and any(
                                    e["mi"] == last["measure_index"]
                                    and e["vi"] == last["voice"]
                                    and e["ei"] == last["event_index"]
                                    for e in s["events"]
                                )
                            ]
                continuations.append(linked[-1]["anchor"] if linked else None)
            else:
                selected.append([0.0] * 8)
                present.append(0)
                continuations.append(None)
                styles.append(None)
        positive = (
            {event["label"]}
            if event["label"] in ("pm", "let")
            else {"pm", "let"}
            if event["label"] == "both"
            else set()
        )
        # Missing visible label/span cannot train a confident semantic answer.
        missing = any(
            not present[i] for i, kind in enumerate(("pm", "let")) if kind in positive
        )
        events.append(
            {
                **event,
                "label": "unknown" if missing else event["label"],
                "geometry": selected,
                "span_present": present,
                "span_styles": styles,
                "continuation_anchors": continuations,
            }
        )
    return {
        **row,
        "events": events,
        "previous_system_image": previous_image(row),
        "previous_system_bbox": previous_row["bbox"] if previous_row else None,
        "span_supervision": "actual painter endpoints and visible text; source ownership retained in measure targets",
    }


def continuation_variant(row, output):
    """Omit redundant reprinted labels only for renderer-linked cross-row spans."""
    from copy import deepcopy

    if not row.get("previous_system_image"):
        return None
    erased = {
        tuple(event["geometry"][kind][4:])
        for event in row["events"]
        for kind in range(2)
        if event["continuation_anchors"][kind] is not None
    }
    if not erased:
        return None
    result = deepcopy(row)
    with Image.open(row["image"]) as raw:
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom in erased:
        draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), fill="white")
    # In a genuine continuation the preceding row cannot retain a terminal
    # hook. Remove exactly the renderer-owned hook, and resume the current
    # dashed line through the omitted label's space using its own paint style.
    # These are explicit engraving interventions in data creation only.
    previous_bbox = row.get("previous_system_bbox")
    if not previous_bbox:
        return None
    with Image.open(row["previous_system_image"]) as raw:
        previous = raw.convert("RGB")
    previous_draw = ImageDraw.Draw(previous)
    source_spans = authored_spans(
        str(Path(row["source_page"]).parent / "geometry.json")
    )
    linked_anchors = {
        tuple(e["continuation_anchors"][kind])
        for e in row["events"]
        for kind in range(2)
        if e["continuation_anchors"][kind] is not None
    }
    for span in source_spans:
        texts = [c for c in span["commands"] if c["type"] == "text"]
        rectangles = [c["bbox"] for c in span["commands"] if c["type"] == "rect"]
        if not texts or not rectangles:
            continue
        prior_anchor = tuple(translated_bbox(texts[0]["bbox"], previous_bbox))
        if prior_anchor in linked_anchors:
            x, y, w, h = rectangles[-1]
            if h > w:
                x -= previous_bbox[0]
                y -= previous_bbox[1]
                previous_draw.rectangle(
                    (x - 1, y - 1, x + w + 1, y + h + 1), fill="white"
                )
        current_anchor = tuple(translated_bbox(texts[0]["bbox"], row["bbox"]))
        if current_anchor in erased:
            horizontal = [b for b in rectangles if b[2] > b[3]]
            if len(horizontal) >= 2:
                x, y, w, h = horizontal[0]
                spacing = horizontal[1][0] - x
                start = (current_anchor[0] + current_anchor[2]) / 2
                x -= row["bbox"][0]
                y -= row["bbox"][1]
                while x - spacing >= start:
                    x -= spacing
                    draw.rectangle((x, y, x + w, y + h), fill="black")
    for event in result["events"]:
        event["anchor_previous"] = [False, False]
        for kind in range(2):
            if event["continuation_anchors"][kind] is not None:
                event["geometry"][kind][4:] = event["continuation_anchors"][kind]
                event["anchor_previous"][kind] = True
    path = output / "continuations" / f"{row['id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    previous_path = path.with_name(path.stem + "-previous.png")
    previous.save(previous_path)
    result.update(
        id=row["id"] + "-continuation",
        image=str(path.resolve()),
        previous_system_image=str(previous_path.resolve()),
        intervention="renderer-owned cross-row continuation: prior terminal hook removed; current redundant label replaced with its own dash style; previous label retained",
    )
    return result
