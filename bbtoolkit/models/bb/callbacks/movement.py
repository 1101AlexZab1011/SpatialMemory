from copy import deepcopy
from typing import Any, Mapping
import numpy as np

from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.dynamics.callbacks.movement import MovementCallback, MovementParameters, MovementSchedulerCallback, TrajectoryCallback
from bbtoolkit.utils.movement import MovementManager


class MentalPositionCallback(BaseCallback):
    """
    A callback designed to update the mental representation of the agent's position and direction during recall in an agent-based learning simulation.

    This callback computes the agent's mental position and direction based on the highest activation rates in place and head direction cells, respectively, during the recall mode. It resets these parameters when not in recall mode.

    Attributes:
        Requires various parameters from the cache to compute and update the mental position and direction.
    """

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating mental position and direction, and initializes the mental movement parameters if not present.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'rates',
            'grid2cart',
            'weights',
            'dynamics_params',
            'mental_movement_params'
        ]

        if 'mental_movement_params' not in cache:
            cache['mental_movement_params'] = MovementParameters()

        super().set_cache(cache, on_repeat)

        self.hd_vector = np.linspace(0, 2*np.pi, len(self.rates.hd))

    def get_coords_from_grid(self, i: int, j: int) -> tuple[float, float]:
        """
        Converts grid indices to Cartesian coordinates.

        Args:
            i (int): The grid row index.
            j (int): The grid column index.

        Returns:
            tuple[float, float]: The Cartesian coordinates corresponding to the grid indices.
        """
        return self.cache.grid2cart[i, j]

    def get_hd_from_rate(self, rate: np.ndarray) -> float:
        """
        Determines the direction based on the highest activation rate in head direction cells.

        Args:
            rate (np.ndarray): The activation rates of head direction cells.

        Returns:
            float: The angle corresponding to the highest activation rate.
        """
        return self.hd_vector[np.argmax(rate)]

    def on_step_end(self, step: int):
        """
        Updates the mental position and direction of the agent at the end of each simulation step, based on the current mode of operation.

        Args:
            step (int): The current step number.
        """
        if self.dynamics_params['mode'] == 'recall':
            x_grid, y_grid = np.unravel_index(np.argmax(
                self.rates.h
            ), self.grid2cart.shape)
            self.mental_movement_params['position'] = self.get_coords_from_grid(x_grid, y_grid)
            self.mental_movement_params['direction'] = self.get_hd_from_rate(self.rates.hd)
        elif self.dynamics_params['mode'] == 'bottom-up':
            self.mental_movement_params['position'] = None
            self.mental_movement_params['direction'] = None


class MentalMovementCallback(MovementCallback):
    """
    A specialized callback for handling mental movement and rotation in an agent-based learning simulation during top-down processing.

    This callback updates the mental representation of the agent's position and direction based on specified movement and rotation targets, simulating cognitive planning and navigation.

    Inherits:
        MovementCallback: For basic movement functionalities.

    Attributes:
        movement_manager (MovementManager): The manager responsible for handling movement computations.
    """
    def __init__(self, movement_manager: MovementManager):
        """
        Initializes the MentalMovementCallback instance with a specified movement manager.

        Args:
            movement_manager (MovementManager): The manager responsible for handling movement computations.
        """
        super().__init__(movement_manager)

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating mental movement parameters, and prepares for movement computations.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'mental_movement_params',
            'dynamics_params',
            'grid2cart'
        ]

        BaseCallback.set_cache(self, cache, on_repeat)
        self.dist = self.movement.distance_per_time(self.dynamics_params.dt)
        self.ang = self.movement.angle_per_time(self.dynamics_params.dt)

    def on_step_begin(self, step: int): # changes position and angle of an agent
        """
        Updates the mental position and direction of the agent at the beginning of each simulation step, based on the current mode of operation and specified targets.

        Args:
            step (int): The current step number.
        """
        if self.dynamics_params.mode == 'top-down':
            if self.mental_movement_params.position is not None and\
                self.mental_movement_params.move_target is not None:
                dist = self.movement.compute_distance(self.mental_movement_params.position, self.mental_movement_params.move_target)
                if dist <= self.dist:
                    self.mental_movement_params.move_target = None

            if self.mental_movement_params.direction is not None and\
                self.mental_movement_params.rotate_target is not None:
                    ang = self.movement.smallest_angle_between(
                        self.mental_movement_params.direction,
                        self.movement.get_angle_with_x_axis(
                            [
                                self.mental_movement_params.rotate_target[0] - self.mental_movement_params.position[0],
                                self.mental_movement_params.rotate_target[1] - self.mental_movement_params.position[1]
                            ]
                        )
                    )
                    if ang <= self.ang:
                        self.mental_movement_params.rotate_target = None

            if self.mental_movement_params.move_target is not None:
                self.mental_movement_params.position = self.move_to_target(
                    self.mental_movement_params.position,
                    self.mental_movement_params.move_target
                )
                self.mental_movement_params.direction = self.rotate_to_target(
                    self.mental_movement_params.position,
                    self.mental_movement_params.direction,
                    self.mental_movement_params.move_target
                )
            elif self.mental_movement_params.rotate_target is not None:
                self.mental_movement_params.direction = self.rotate_to_target(
                    self.mental_movement_params.position,
                    self.mental_movement_params.direction,
                    self.mental_movement_params.rotate_target
                )


class MentalMovementSchedulerCallback(MovementSchedulerCallback):
    """
    A specialized callback for scheduling mental movements in an agent-based learning simulation.

    This callback manages a queue of positions that represent the agent's planned trajectory during cognitive navigation tasks. It updates the agent's next movement target based on this schedule.

    Inherits:
        MovementSchedulerCallback: For basic movement scheduling functionalities.

    Attributes:
        Requires various parameters from the cache to manage and update the mental movement schedule and trajectory.
    """
    def set_cache(self, cache: Any, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the mental movement schedule and trajectory based on planned positions, and prepares for movement scheduling.

        Args:
            cache (Any): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'mental_movement_params',
            'mental_movement_schedule',
            'mental_trajectory'
        ]
        cache['mental_movement_schedule'] = self.positions
        cache['mental_trajectory'] = deepcopy(self.positions)
        BaseCallback.set_cache(self, cache, on_repeat)

    def on_step_end(self, step: int):
        """
        Updates the mental movement target at the end of each simulation step, based on the current schedule of planned movements.

        Args:
            step (int): The current step number.
        """
        if len(self.mental_movement_schedule):
            if self.mental_movement_params.move_target is None:
                self.mental_movement_params.move_target = self.mental_movement_schedule.pop(0)


class MentalTrajectoryCallback(TrajectoryCallback):
    """
    A specialized callback for managing the mental trajectory of an agent in an agent-based learning simulation.

    This callback updates the agent's mental trajectory and movement schedule based on its current position, movement target, and direction. It ensures that the mental trajectory reflects the planned path towards the target.

    Inherits:
        TrajectoryCallback: For basic trajectory functionalities.

    Attributes:
        Requires various parameters from the cache to manage and update the mental trajectory and movement schedule.
    """
    def set_cache(self, cache: Any, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the mental trajectory if not present, and prepares for trajectory management.

        Args:
            cache (Any): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'mental_movement_params',
            'mental_movement_schedule',
            'mental_trajectory'
        ]

        if 'mental_trajectory' not in cache:
            cache['mental_trajectory'] = list()

        BaseCallback.set_cache(self, cache, on_repeat)

    def on_step_begin(self, step: int):
        """
        Updates the mental trajectory and movement schedule at the beginning of each simulation step, based on the agent's current position, movement target, and direction.

        Args:
            step (int): The current step number.
        """
        if self.mental_movement_params.move_target is not None:
            if self.mental_movement_params.move_target not in self.mental_trajectory:
                xy = self.trajectory_manager(
                    self.mental_movement_params.position,
                    self.mental_movement_params.move_target,
                    self.mental_movement_params.direction
                )
                self.mental_trajectory.clear()
                self.mental_trajectory += [tuple(item) for item in xy.tolist()]
                self.mental_movement_schedule.clear()
                self.mental_movement_schedule += deepcopy(self.mental_trajectory)
                self.mental_movement_params.move_target = self.mental_movement_schedule.pop(0)


