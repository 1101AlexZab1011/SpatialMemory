
from typing import Mapping

import numpy as np

from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.dynamics.callbacks.movement import MovementParameters


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


class FramesStoringCallback(BaseCallback):
    """
    A callback designed to store frames of the simulation at a specified rate.

    This callback saves snapshots of the simulation's current state, allowing for the creation of visualizations such as animations or time-lapse videos.

    Attributes:
        save_rate (int): The rate at which frames are saved (e.g., every `save_rate` steps).
        savedir (str): The directory where the frames are saved.
    """
    def __init__(self, save_rate: int, savedir: str):
        """
        Initializes the FramesStoringCallback instance with a specified save rate and directory.

        Args:
            save_rate (int): The rate at which frames are saved.
            savedir (str): The directory where the frames are saved.
        """
        super().__init__()
        self.save_rate = save_rate
        self.savedir = savedir

    def set_cache(
        self,
        cache: Mapping,
        on_repeat: str = 'raise'
    ):
        """
        Sets the cache with the provided mapping and specifies the required cache keys for saving frames.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['fig', 'dynamics_params']
        super().set_cache(cache, on_repeat)

    def on_step_end(self, step: int):
        """
        Executes the logic for saving a frame at the end of a simulation step, based on the specified save rate.

        Args:
            step (int): The current step number.
        """
        if self.dynamics_params.step % self.save_rate == 0:
            self.fig.savefig(f'{self.savedir}/frame_{self.dynamics_params.step}.png')
