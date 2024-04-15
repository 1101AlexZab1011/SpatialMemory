from typing import Any, Callable, Generator, Mapping

from bbtoolkit.data import Copyable, WritablePickle
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.structures import BaseCallbacksManager, CallbacksCollection


class DynamicsManager(BaseCallbacksManager, WritablePickle, Copyable):
    """
    Manages the dynamics of a system by coordinating callbacks and maintaining a cache for shared data.
    This manager allows for the execution of callbacks at specific steps and cycles during a simulation.

    Attributes:
        steps_per_cycle (int): The number of steps in each cycle, determined by the inverse of the time step (dt).
        timer (int): A counter to keep track of the current step within the simulation.
        callbacks (CallbacksCollection): A collection of callbacks to be executed during the simulation.
        cache (dict): A shared data cache accessible by all callbacks.

    Args:
        dt (int): The time step of the simulation. Determines the frequency of callback execution.
        callbacks (list[BaseCallback], optional): An initial list of callbacks to be included in the simulation.
        cache (Mapping, optional): An initial cache of data to be shared among callbacks.

    Methods:
        add_callback(callback: BaseCallback):
            Adds a new callback to the collection and validates its requirements.

        remove_callback(index: int):
            Removes a callback from the collection by its index and cleans up the cache.

        _step():
            Executes a single step of the simulation, triggering the appropriate callbacks.

        run(n_steps: int):
            Runs the simulation for a specified number of steps.

        __call__(time: float) -> Generator[Any, None, None]:
            Runs the simulation for a specified amount of time, yielding control after each cycle.
    """
    def __init__(self, dt: int, callbacks: list[BaseCallback] = None, cache: Mapping = None):
        """
        Initializes the DynamicsManager with a time step, an optional list of callbacks, and an optional cache.
        """
        super().__init__(callbacks, cache)
        self.steps_per_cycle = int(1/dt)
        self.timer = 0

    def _step(self):
        """
        Executes a single step of the simulation, triggering the appropriate callbacks.
        """
        if not self.timer%self.steps_per_cycle: # only if new cycle is started
            self.callbacks.execute('on_cycle_begin', self.timer)

        self.callbacks.execute('on_step_begin', self.timer%self.steps_per_cycle)
        self.timer += 1
        self.callbacks.execute('on_step_end', self.timer%self.steps_per_cycle)

        if self.timer%self.steps_per_cycle == self.steps_per_cycle - 1: # only of cycle is finished
            self.callbacks.execute('on_cycle_end', self.timer)

    def run(self, n_steps: int):
        """
        Runs the simulation for a specified number of steps.

        Args:
            n_steps (int): The number of steps to run the simulation for.

        Returns:
            The result of the 'on_iteration_end' callback execution.
        """
        self.callbacks.execute('on_iteration_begin', n_steps)

        for _ in range(n_steps):
            self._step()

        return self.callbacks.execute('on_iteration_end', self.timer/self.steps_per_cycle)

    def __call__(self, time: float | bool | Callable[[Any], bool]) -> Generator[Any, None, None]:
        """
        Runs the simulation for a specified amount of time, yielding control after each cycle.

        Args:
            time (float): The total time to run the simulation for.
            tine (float | bool | Callable[[Any], bool]): The total time to run the simulation for.
                If a float, the simulation will run for the specified time.
                If a bool, the simulation will run indefinitely (if the condition is True).
                If a Callable, the simulation will run until the condition is False.
                The Callable should accept the result of each cycle's execution as an argument.

        Yields:
            The result of each cycle's execution during the simulation.
        """
        if isinstance(time, bool):
            while time:
                yield self.run(self.steps_per_cycle)
        elif isinstance(time, (int, float)):
            rest = int(time*self.steps_per_cycle%self.steps_per_cycle)
            rest = [rest] if rest > 0 else []
            cycles = [self.steps_per_cycle for _ in range(int(time))] + rest

            self.callbacks.execute('on_simulation_begin', len(cycles))

            for cycle in cycles:
                yield self.run(cycle)

            self.callbacks.execute('on_simulation_end')
        elif isinstance(time, Callable):
            run = True
            while run:
                out = self.run(self.steps_per_cycle)
                run = time(out)
                yield out