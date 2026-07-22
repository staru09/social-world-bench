import textwrap

import pytest

from swb.world.constitution import ConstitutionError, load_constitution, resolve_world


def test_loads_generic_social_v1():
    path = resolve_world("generic_social_v1")
    c = load_constitution(path)
    assert c.world_id == "generic_social_v1"
    assert {p.id for p in c.peers} == {"alex", "sam", "jordan"}
    assert c.rules.noise_ratio == 0.65


def test_rejects_relationship_to_unknown_peer(tmp_path):
    bad = tmp_path / "constitution.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            world_id: bad
            peers:
              - {id: a, name: A, persona: x}
            relationships:
              - {from_peer: a, to_peer: ghost, type: friend, strength: 1.0}
            """
        )
    )
    with pytest.raises(ConstitutionError):
        load_constitution(bad)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(ConstitutionError):
        load_constitution(tmp_path / "nope.yaml")
