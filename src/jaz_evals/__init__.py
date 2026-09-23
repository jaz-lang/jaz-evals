"""Evals for benchmarking JAZ.

Importing this package registers the built-in environments and harnesses, so a config that names one
resolves without the caller importing it first.
"""

from jaz_evals.config import ConfigError, EnvConfig, EvalConfig, MethodConfig, load_eval_config
from jaz_evals.env import AgentEnv, Env, EnvAccessError, Grade, ToolSpec, internal, root_only
from jaz_evals.eval_harness import (
    AttemptRecord,
    aggregate_records,
    attempt_dir,
    run_attempt,
    run_dir,
    run_evaluation,
)
from jaz_evals.harness import Harness, RunReport, Usage
from jaz_evals.isolation import Isolation, IsolationError
from jaz_evals.provenance import write_provenance
from jaz_evals.registry import (
    ComponentNameError,
    UnknownComponentError,
    get_env,
    get_harness,
    register_env,
    register_harness,
    registered_envs,
    registered_harnesses,
)
from jaz_evals.run_id import check_run_id, new_run_id

__all__ = [
    "AgentEnv",
    "AttemptRecord",
    "ComponentNameError",
    "ConfigError",
    "Env",
    "EnvAccessError",
    "EnvConfig",
    "EvalConfig",
    "Grade",
    "Harness",
    "Isolation",
    "IsolationError",
    "MethodConfig",
    "RunReport",
    "ToolSpec",
    "UnknownComponentError",
    "Usage",
    "aggregate_records",
    "attempt_dir",
    "check_run_id",
    "get_env",
    "get_harness",
    "internal",
    "load_eval_config",
    "new_run_id",
    "register_env",
    "register_harness",
    "registered_envs",
    "registered_harnesses",
    "root_only",
    "run_attempt",
    "run_dir",
    "run_evaluation",
    "write_provenance",
]

# Imported for the registration side effect, after the names above exist.
from jaz_evals import envs, harnesses

__all__ += ["envs", "harnesses"]
