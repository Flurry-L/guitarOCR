"""Shared Rust observation contract and assembly, exposed to research."""

import json

from scorelib import _native

LAYOUT_FIELDS = ("page", "source_page", "system_index")
NOTE_RELATIONS = ("tie", "hammer", "sl", "ss")


def observation_schema():
    return json.loads(_native.observation_schema())


def parse_observation(value):
    return json.loads(_native.parse_observation(json.dumps(value)))


def range_relation(relation):
    return _native.range_relation(json.dumps(relation))


def relation_voice(relation):
    return _native.relation_voice(json.dumps(relation))


def relation_problem(observation, relation):
    return _native.relation_problem(json.dumps(observation), json.dumps(relation))


def assemble(records, *, auxiliary=None):
    return json.loads(_native.assemble_observations(
        json.dumps(list(records)),
        json.dumps(list(auxiliary)) if auxiliary is not None else None,
    ))
