from __future__ import annotations

from pathlib import Path

import yaml

from swb.schema import Constitution

WORLDS_DIR = Path(__file__).resolve().parents[3] / "worlds"


class ConstitutionError(ValueError):
    """Raised when a constitution is structurally invalid or self-inconsistent."""


def load_constitution(path: str | Path) -> Constitution:
    """Parse and validate a constitution YAML file.

    Pydantic covers field-level validation; we add cross-field checks that the
    relationship graph and interaction rules only reference declared peers.
    """
    path = Path(path)
    if not path.exists():
        raise ConstitutionError(f"constitution not found: {path}")
    data = yaml.safe_load(path.read_text())
    try:
        constitution = Constitution.model_validate(data)
    except Exception as e:  # pydantic ValidationError
        raise ConstitutionError(f"invalid constitution {path}: {e}") from e

    ids = constitution.peer_ids()
    if len(ids) != len(constitution.peers):
        raise ConstitutionError("duplicate peer ids in constitution")
    for r in constitution.relationships:
        if r.from_peer not in ids:
            raise ConstitutionError(f"relationship references unknown peer: {r.from_peer}")
        if r.to_peer not in ids:
            raise ConstitutionError(f"relationship references unknown peer: {r.to_peer}")
    if constitution.rules.allowed_speakers is not None:
        for pid in constitution.rules.allowed_speakers:
            if pid not in ids:
                raise ConstitutionError(f"allowed_speakers references unknown peer: {pid}")
    return constitution


def resolve_world(world: str | Path) -> Path:
    """Resolve a world name (e.g. 'generic_social_v1') or a path to its constitution."""
    p = Path(world)
    if p.exists() and p.is_file():
        return p
    candidate = WORLDS_DIR / str(world) / "constitution.yaml"
    if candidate.exists():
        return candidate
    raise ConstitutionError(f"cannot resolve world {world!r} (looked for {candidate})")
