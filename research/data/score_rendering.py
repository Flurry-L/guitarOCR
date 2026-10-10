"""Render authored alphaTab scores and check their musical targets."""

import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from scorelib.pitch_context import convert_pitch_target
from scorelib.score import duration_ticks, parse_measure_target, validate_measure_target

from research.defaults import PROJECT_ROOT


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


@contextmanager
def browser_page():
    from playwright.sync_api import sync_playwright

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(PROJECT_ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=['--no-sandbox'])
            try:
                page = browser.new_page(viewport={'width': 1900, 'height': 1800}, device_scale_factor=1)
                yield page, f'http://127.0.0.1:{server.server_port}'
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def audit(row):
    measure = parse_measure_target(row["target"])
    if row.get('visual_pitch') and row['mode'] == 'both' and row.get('instrument') != 'drums':
        if not row.get('pitch_context'):
            return ['missing_written_pitch_context']
        if row.get('sounding_target'):
            reference = convert_pitch_target(row['sounding_target'], row['pitch_context'],
                                             to_written=True, mode='both')
            expected = parse_measure_target(reference)
            if (measure['voices'] != expected['voices']
                    or measure.get('key_signature') != expected.get('key_signature')):
                return ['written_pitch_contract_mismatch']
    # Written pitch and TAB are independent observations; validate rhythm and
    # string geometry without falsely treating written guitar pitches as sound.
    _, errors = validate_measure_target(row["target"], row["mode"], string_count=len(row.get("tuning", [])) or None)
    if errors:
        return errors
    beats, denominator = map(int, row["score_state"]["time"].split("/"))
    length = beats * 3840 // denominator
    for voice in measure["voices"]:
        end = 0
        for event in voice["events"]:
            if event["start"] != end:
                return ["noncontiguous_voice"]
            end += duration_ticks(event["duration"])
            for note in event["notes"]:
                effects = set(event.get("effects", [])) | set(note.get("effects", []))
                if {"pm", "let"} <= effects:
                    return ["contradictory_pm_and_let"]
                if isinstance(note.get("fret"), int) and not 0 <= note["fret"] <= 24:
                    return ["fret_outside_practical_range"]
        if end != length:
            return ["incomplete_voice"]
    return []
