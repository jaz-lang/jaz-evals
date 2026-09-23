"""Name-to-class lookup for environments and harnesses.

A config names its env and method as strings (`env.name`, `method.name`); the eval harness resolves those
to classes here. Registering by decorator keeps the mapping next to the class it names, so adding an env
or a harness does not mean editing a central table.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from jaz_evals.env import Env
from jaz_evals.harness import Harness

# Registered names are config keys *and* path components, so they are held to what is safe as both.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

_ENVS: dict[str, type[Env]] = {}
_HARNESSES: dict[str, type[Harness]] = {}


class ComponentNameError(ValueError):
    """Raised when an environment or harness is registered under an unusable name."""


class UnknownComponentError(KeyError):
    """Raised when a config names an environment or a harness that is not registered."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


def register_env[E: type[Env]](name: str) -> Callable[[E], E]:
    """Register an environment class under `name`, for use as `env.name` in a config."""

    def decorate(cls: E) -> E:
        _register(_ENVS, name, cls, "environment")
        return cls

    return decorate


def register_harness[H: type[Harness]](name: str) -> Callable[[H], H]:
    """Register a harness class under `name`, for use as `method.name` in a config."""

    def decorate(cls: H) -> H:
        _register(_HARNESSES, name, cls, "harness")
        return cls

    return decorate


def get_env(name: str) -> type[Env]:
    """Return the environment class registered under `name`."""
    return _lookup(_ENVS, name, "environment")


def get_harness(name: str) -> type[Harness]:
    """Return the harness class registered under `name`."""
    return _lookup(_HARNESSES, name, "harness")


def registered_envs() -> list[str]:
    """Return the registered environment names."""
    return sorted(_ENVS)


def registered_harnesses() -> list[str]:
    """Return the registered harness names."""
    return sorted(_HARNESSES)


def _register[T: type](registry: dict[str, T], name: str, cls: T, kind: str) -> None:
    if not _NAME_PATTERN.match(name):
        raise ComponentNameError(
            f"invalid {kind} name {name!r}: it becomes a directory name, so it must be alphanumerics, "
            "'-' or '_', starting with an alphanumeric"
        )
    existing = registry.get(name)
    if existing is not None and existing is not cls:
        raise UnknownComponentError(f"{kind} name {name!r} is already registered to {existing.__name__}")
    registry[name] = cls


def _lookup[T: type](registry: dict[str, T], name: str, kind: str) -> T:
    try:
        return registry[name]
    except KeyError:
        known = ", ".join(sorted(registry)) or "(none registered)"
        raise UnknownComponentError(f"unknown {kind} {name!r}; registered: {known}") from None
