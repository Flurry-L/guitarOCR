"""Metadata prompts, schemas and production interpretation shared with Rust."""

import json
from scorelib import _native


def prompt(kind):
    return _native.information_prompt(kind)


def schema(kind):
    return json.loads(_native.information_schema(kind))


def parse_response(raw, kind):
    return json.loads(_native.parse_information(raw, kind))
