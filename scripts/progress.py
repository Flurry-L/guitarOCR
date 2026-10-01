"""Concise terminal progress for the Python installer and model downloads."""


def progress(label, *, detail='', completed=None, total=None):
    if completed is None or completed == total:
        print(f'{label}：{detail}' if detail else label, flush=True)
