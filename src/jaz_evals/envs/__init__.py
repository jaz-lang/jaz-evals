"""Environment implementations.

Importing this package registers every environment, so a config can name one as `env.name`. Each env
module imports its benchmark's heavy dependencies lazily, so importing this package does not require any
of them (StuLifeEnv's `stulife` package and AppWorldEnv's `appworld` included).
"""

from jaz_evals.envs.appworld import AppWorldEnv
from jaz_evals.envs.stulife import StuLifeEnv
from jaz_evals.registry import register_env

register_env("appworld")(AppWorldEnv)
register_env("stulife")(StuLifeEnv)

__all__ = ["AppWorldEnv", "StuLifeEnv"]
