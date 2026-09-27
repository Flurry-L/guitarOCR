"""Installation progress shared by the terminal launcher and desktop window."""

import json
import os


def progress(stage, label, *, detail='', completed=None, total=None):
    if os.environ.get('GUITAROCR_DESKTOP'):
        print('GUITAROCR_PROGRESS ' + json.dumps(
            dict(stage=stage, label=label, detail=detail, completed=completed, total=total),
            ensure_ascii=False), flush=True)
    elif completed is None or completed == total:
        print(label, flush=True)
