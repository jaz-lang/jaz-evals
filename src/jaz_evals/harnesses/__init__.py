"""Method harness implementations.

Importing this package registers every harness, so a config can name one as `method.name`. The
harnesses import their method's library lazily, so importing this package does not require any of
them to be installed.
"""

from jaz_evals.harnesses.ace import AceHarness
from jaz_evals.harnesses.jaz_harness import JazHarness
from jaz_evals.harnesses.jaz_per_task import JazPerTaskHarness
from jaz_evals.harnesses.letta_harness import LettaHarness
from jaz_evals.harnesses.smolagents_harness import SmolagentsHarness
from jaz_evals.registry import register_harness

register_harness("ace")(AceHarness)
register_harness("jaz")(JazHarness)
register_harness("jaz_per_task")(JazPerTaskHarness)
register_harness("smolagents")(SmolagentsHarness)
register_harness("letta")(LettaHarness)

__all__ = ["AceHarness", "JazHarness", "JazPerTaskHarness", "LettaHarness", "SmolagentsHarness"]
