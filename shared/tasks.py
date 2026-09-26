"""Cooperative cancellation between model generations."""


class Cancelled(RuntimeError):
    pass
