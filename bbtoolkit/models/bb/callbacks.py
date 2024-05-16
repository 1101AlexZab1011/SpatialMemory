
from typing import Mapping

from bbtoolkit.dynamics.callbacks import BaseCallback


class TimerCallback(BaseCallback):
    """
    A callback designed to keep track of simulation steps within an agent-based learning simulation.

    This callback updates a step counter in the dynamics parameters each time a simulation step ends, providing a mechanism to monitor the progress of the simulation.

    Inherits:
        BaseCallback: For basic callback functionality.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and initializes the step counter in the dynamics parameters.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['dynamics_params']
        cache.dynamics_params.step = 0
        super().set_cache(cache, on_repeat)

    def on_step_end(self, step: int):
        """
        Increments the step counter in the dynamics parameters at the end of each simulation step.

        Args:
            step (int): The current step number.
        """
        self.dynamics_params.step += 1



