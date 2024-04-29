from typing import Mapping

from bbtoolkit.structures import BaseCallback as _BaseCallback


class BaseCallback(_BaseCallback):
    """
    Base class for creating callbacks that can be used during a simulation or iterative process.

    This class provides a basic structure for implementing callbacks with customizable actions at different stages of a simulation or iterative process. It includes methods that are called at the beginning and end of cycles, steps, iterations, and the entire simulation.

    Attributes:
        _cache (Mapping, optional): A cache for storing temporary data during the callback's lifecycle. Defaults to None.
        _requires (list): A list of requirements or dependencies needed by the callback. Defaults to an empty list.

    Properties:
        cache: Returns the current cache.
        requires: Returns the current list of requirements.

    Methods:
        set_cache(cache: Mapping): Sets the cache with the provided mapping.
        on_cycle_begin(total_steps: int): Called at the beginning of a cycle.
        on_cycle_end(total_steps: int): Called at the end of a cycle.
        on_step_begin(step: int): Called at the beginning of a step.
        on_step_end(step: int): Called at the end of a step.
        on_iteration_begin(n_steps: int): Called at the beginning of an iteration.
        on_iteration_end(n_cycles_passed: int): Called at the end of an iteration.
        on_simulation_begin(n_iterations): Called at the beginning of the simulation.
        on_simulation_end(): Called at the end of the simulation.
    """

    def on_cycle_begin(self, total_steps: int):
        """
        Called at the beginning of a cycle.

        Args:
            total_steps (int): The total number of steps in the current cycle.
        """
        pass

    def on_cycle_end(self, total_steps: int):
        """
        Called at the end of a cycle.

        Args:
            total_steps (int): The total number of steps in the current cycle.
        """
        pass

    def on_step_begin(self, step: int):
        """
        Called at the beginning of a step.

        Args:
            step (int): The current step number.
        """
        pass

    def on_step_end(self, step: int):
        """
        Called at the end of a step.

        Args:
            step (int): The current step number.
        """
        pass

    def on_iteration_begin(self, n_steps: int):
        """
        Called at the beginning of an iteration.

        Args:
            n_steps (int): The number of steps in the current iteration.
        """
        pass

    def on_iteration_end(self, n_cycles_passed: int):
        """
        Called at the end of an iteration.

        Args:
            n_cycles_passed (int): The number of cycles that have passed in the current iteration.
        """
        pass

    def on_simulation_begin(self, n_iterations):
        """
        Called at the beginning of the simulation.

        Args:
            n_iterations: The number of iterations in the simulation.
        """
        pass

    def on_simulation_end(self):
        """
        Called at the end of the simulation.
        """
        pass

    def on_load(self, **kwargs):
        """
        Called when the callbacks manager is loaded from a serialized state.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def on_copy(self, **kwargs):
        """
        Called when the callbacks manager is copied.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass
