"""Loading of the per-pair YAML eval configs.

One config file describes one environment-method pair:

    method:
      name: jaz
      prompt_path: ../prompts/long_horizon/jaz.md
      config:
        ...
    env:
      name: stulife
      config:
        ...

`method.config` and `env.config` are left opaque here -- the harness for `method.name` and the env class for
`env.name` interpret their own contents. Only the envelope is validated, so an unknown or misspelled top-level
key fails at load rather than being silently ignored.

`method.prompt_path` is the domain-method prompt: technique that belongs to running *this method* on *this
domain* and to neither alone -- for JAZ on long-horizon, how to search REPL history and how to hand off to a
subagent before the session ends. It is a config key because a (domain, method) pair has no class to hang it
on, unlike an env, whose instructions are `Env.get_instructions` and are not configured at all.

`prompt_path` is resolved relative to the config file's directory, so a config and its prompts move together.
It is a top-level `method` key rather than one inside `method.config` because only the loader knows where the
config file lives, and a path inside the opaque mapping would reach the harness unresolved. It is optional: a
method that needs no technique for the domain omits it, and `method.prompt_path` is `None`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or carries unexpected keys."""


@dataclass(frozen=True)
class MethodConfig:
    """The method under test, its domain-method prompt, and its settings.

    `prompt_path` is `None` when the pairing ships no domain-method prompt -- a method whose technique
    needs no guidance for the domain (whose model handles growing context itself, so
    there is no long-horizon technique to teach). The harness then passes the agent no `guidance`.
    """

    name: str
    prompt_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class EnvConfig:
    """The environment and the settings passed to its constructor."""

    name: str
    config: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class EvalConfig:
    """One environment-method pair.

    `source_text` is the config file's text exactly as it was read, and is what a run records as the
    config it ran; it is `None` for a config built in code rather than loaded from a file.
    """

    # The text is kept here rather than re-read from `source` when a run starts, because those answer
    # different questions: re-reading says what the file contains *now*. A sweep driver that loads its
    # configs up front can run for hours while `configs/` is edited, a generated config's temp dir can
    # be cleaned up, and a relative `source` silently retargets once a harness chdirs into a task
    # workspace. Capturing at load costs one field and removes all three.
    #
    # Re-serialising this dataclass is never an acceptable stand-in: `prompt_path` has been resolved to
    # an absolute path and every comment is gone, so the output would be a file that is *not* the config
    # that ran. `repr=False` also keeps a whole YAML file out of any error quoting this object.

    method: MethodConfig
    env: EnvConfig
    source: Path | None = None
    source_text: str | None = field(default=None, repr=False)


def load_eval_config(path: Path | str) -> EvalConfig:
    """Load and validate the eval config at `path`."""
    source = Path(path)
    try:
        text = source.read_text()
    except FileNotFoundError as exc:
        raise ConfigError(f"{source}: no such config file") from exc
    except OSError as exc:
        raise ConfigError(f"{source}: cannot read config file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{source}: config file is not valid UTF-8: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc

    document = _as_mapping(raw, source, "document")
    _reject_unknown(document, {"method", "env"}, source, "document")

    return EvalConfig(
        method=_parse_method(_required(document, "method", source), source),
        env=_parse_env(_required(document, "env", source), source),
        source=source,
        source_text=text,
    )


def _parse_method(raw: Any, source: Path) -> MethodConfig:
    section = _as_mapping(raw, source, "method")
    _reject_unknown(section, {"name", "prompt_path", "config"}, source, "method")
    # Optional: a pairing whose method needs no domain guidance omits it entirely,
    # rather than shipping an empty file that would still render a blank guidance block. When present,
    # it is resolved and its existence checked here, so a typo'd path fails at load rather than at run.
    prompt_path: Path | None = None
    declared = section.get("prompt_path")
    if declared is not None:
        resolved = (source.parent / _as_str(declared, source, "method.prompt_path")).resolve()
        if not resolved.is_file():
            raise ConfigError(f"{source}: method.prompt_path is not a file: {resolved}")
        prompt_path = resolved
    return MethodConfig(
        name=_as_str(_required(section, "name", source, "method"), source, "method.name"),
        prompt_path=prompt_path,
        config=_optional_mapping(section, "config", source, "method.config"),
    )


def _parse_env(raw: Any, source: Path) -> EnvConfig:
    section = _as_mapping(raw, source, "env")
    _reject_unknown(section, {"name", "config"}, source, "env")
    return EnvConfig(
        name=_as_str(_required(section, "name", source, "env"), source, "env.name"),
        config=_optional_mapping(section, "config", source, "env.config"),
    )


def _required(section: dict[str, Any], key: str, source: Path, where: str = "document") -> Any:
    if key not in section or section[key] is None:
        raise ConfigError(f"{source}: {where} is missing required key {key!r}")
    return section[key]


def _optional_mapping(section: dict[str, Any], key: str, source: Path, where: str) -> dict[str, Any]:
    """Return `section[key]` as a mapping, defaulting to empty when absent.

    Absent and wrong-typed are kept distinct: an omitted `config:` defaults, but `config: []` or
    `config: 0` is a malformed config and is reported rather than quietly treated as empty.
    """
    raw = section.get(key)
    return {} if raw is None else _as_mapping(raw, source, where)


def _reject_unknown(section: dict[str, Any], allowed: set[str], source: Path, where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"{source}: {where} has unexpected key(s) {', '.join(repr(k) for k in unknown)}; "
            f"expected only {', '.join(repr(k) for k in sorted(allowed))}"
        )


def _as_mapping(value: Any, source: Path, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: {where} must be a mapping, got {type(value).__name__}")
    return cast(dict[str, Any], value)


def _as_str(value: Any, source: Path, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{source}: {where} must be a string, got {type(value).__name__}")
    return value
