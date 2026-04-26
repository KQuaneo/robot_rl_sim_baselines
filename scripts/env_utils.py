from __future__ import annotations

import re

import gymnasium as gym
from gymnasium.envs.registration import registry


def resolve_env_id(env_id: str) -> str:
    """Return env_id if available, otherwise fall back to the highest local version."""
    try:
        gym.spec(env_id)
        return env_id
    except Exception:
        pass

    match = re.fullmatch(r"(.+)-v\d+", env_id)
    if not match:
        raise ValueError(f"Gymnasium environment {env_id!r} is not installed")

    prefix = match.group(1)
    versions = []
    for registered_id in registry.keys():
        version_match = re.fullmatch(rf"{re.escape(prefix)}-v(\d+)", registered_id)
        if version_match:
            versions.append((int(version_match.group(1)), registered_id))

    if not versions:
        raise ValueError(f"No installed Gymnasium environment matches {env_id!r}")

    resolved = max(versions)[1]
    print(f"Environment {env_id!r} is unavailable locally; using {resolved!r}.")
    return resolved
