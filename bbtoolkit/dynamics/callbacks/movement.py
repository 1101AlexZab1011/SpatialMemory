from copy import deepcopy
import math
from typing import Any, Mapping
from bbtoolkit.dynamics.callbacks import BaseCallback

from bbtoolkit.movement import MovementManager
from bbtoolkit.movement.trajectory import TrajectoryManager


class MovementCallback(BaseCallback):
    """
    A callback class designed to manage the movement and rotation of an agent within a simulation environment.

    This callback integrates with a MovementManager instance to calculate and update the agent's position and direction based on
    specified targets for movement and rotation. It utilizes the simulation's time step (dt) to determine the distance
    and angle the agent can move or rotate within a single step.

    Attributes:
        dt (float): The time step of the simulation.
        movement (MovementManager): An instance of MovementManager to manage calculations related to movement and rotation.
        dist (float): The maximum distance the agent can move in one time step.
        ang (float): The maximum angle the agent can rotate in one time step.

    Args:
        dt (float): The time step of the simulation.
        movement_manager (MovementManager): An instance of MovementManager.

    Methods:
        set_cache(cache: Mapping):
            Sets the cache for the callback and initializes required keys.

        rotate_to_target(position: tuple[float, float], direction: float, target: tuple[float, float]) -> float:
            Calculates the new direction after rotating towards a target within the constraints of the maximum rotation angle.

        move_to_target() -> tuple[float, float]:
            Calculates the new position after moving towards the move target within the constraints of the maximum distance.

        on_step_begin(step: int):
            Updates the agent's position and direction at the beginning of each simulation step based on the current targets.

    Notes:
        This callback requires the following keys in the cache:
            - 'position': The current position of the agent.
            - 'direction': The current direction of the agent in radians.
            - 'move_target': The target position for movement.
            - 'rotate_target': The target position for rotation.

    """
    def __init__(self, movement_manager: MovementManager):
        """
        Initializes the MovementCallback with a time step and a MovementManager instance.
        """
        super().__init__()
        self.movement = movement_manager
        self.dt = None
        self.dist = None
        self.ang = None

    def set_cache(self, cache: Mapping):
        """
        Sets the cache for the callback and initializes required keys for movement and rotation targets.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)
        self.cache['position'] = self.movement.position
        self.cache['direction'] = self.movement.direction
        self.cache['move_target'] = None
        self.cache['rotate_target'] = None
        self.dt = self.cache['dt']
        self.dist = self.movement.distance_per_time(self.dt)
        self.ang = self.movement.angle_per_time(self.dt)
        self.requires = ['dt', 'position', 'direction', 'move_target', 'rotate_target']

    def rotate_to_target(self, position: tuple[float, float], direction: float, target: tuple[float, float]) -> float:
        """
        Calculates the new direction after rotating towards a target within the constraints of the maximum rotation angle.

        Args:
            position (tuple[float, float]): The current position of the agent.
            direction (float): The current direction of the agent in radians.
            target (tuple[float, float]): The target position to rotate towards.

        Returns:
            float: The new direction of the agent after rotating towards the target.
        """
        angle_to_target = math.atan2(
            target[1] - position[1],
            target[0] - position[0]
        )
        angle_to_target = (angle_to_target + 2 * math.pi) % (2 * math.pi)
        angle_diff = angle_to_target - direction
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        rotation = min(abs(angle_diff), self.ang) * math.copysign(1, angle_diff)
        return (direction + rotation) % (2 * math.pi)

    def move_to_target(self) -> tuple[float, float]:
        """
        Calculates the new position after moving towards the move target within the constraints of the maximum distance.

        Returns:
            tuple[float, float]: The new position of the agent after moving towards the move target.
        """
        #! polar coords with angle as a direction gives more plausible movement for long distances
        # return position[0] + self.dist * math.cos(direction),\
        #     position[1] + self.dist * math.sin(direction)
        ang = self.movement.get_angle_with_x_axis(
            [
                self.cache['move_target'][0] - self.cache['position'][0],
                self.cache['move_target'][1] - self.cache['position'][1]
            ]
        )
        return self.cache['position'][0] + self.dist * math.cos(ang),\
            self.cache['position'][1] + self.dist * math.sin(ang)

    def on_step_begin(self, step: int): # changes position and angle of an agent
        """
        Updates the agent's position and direction at the beginning of each simulation step based on the current targets.

        Args:
            step (int): The current step of the simulation.
        """
        if self.cache['position'] is not None and\
            self.cache['move_target'] is not None:
            dist = self.movement.compute_distance(self.cache['position'], self.cache['move_target'])
            if dist <= self.dist:
                self.cache['move_target'] = None

        if self.cache['direction'] is not None and\
            self.cache['rotate_target'] is not None:
                ang = self.movement.smallest_angle_between(
                    self.cache['direction'],
                    self.movement.get_angle_with_x_axis(
                        [
                            self.cache['rotate_target'][0] - self.cache['position'][0],
                            self.cache['rotate_target'][1] - self.cache['position'][1]
                        ]
                    )
                )
                if ang <= self.ang:
                    self.cache['rotate_target'] = None

        if self.cache['move_target'] is not None:
            self.cache['position'] = self.move_to_target()
            self.cache['direction'] = self.rotate_to_target(self.cache['position'], self.cache['direction'], self.cache['move_target'])
        elif self.cache['rotate_target'] is not None:
            self.cache['direction'] = self.rotate_to_target(self.cache['position'], self.cache['direction'], self.cache['rotate_target'])


class MovementSchedulerCallback(BaseCallback):
    """
    A callback class designed to manage the movement of an agent through a predefined sequence of positions.

    This callback allows for the scheduling of an agent's movement through a list of specified positions. At each step
    of the simulation, if the agent does not have a current movement target, the next position in the schedule is set
    as the target. This facilitates the creation of complex movement patterns or paths for the agent to follow.

    Attributes:
        positions (list[tuple[float, float]]): A list of positions (as tuples of floats) through which the agent is scheduled to move.

    Args:
        positions (list[tuple[float, float]], optional): An optional list of positions for the initial movement schedule. Defaults to None.

    Methods:
        set_cache(cache: Any):
            Sets the cache for the callback and initializes required keys for managing the movement schedule.

        on_step_end(step: int):
            Updates the agent's movement target at the end of each simulation step, based on the movement schedule.

    Notes:
        This callback requires the following keys in the cache:
            - 'position': The current position of the agent.
            - 'move_target': The target position for movement.
            - 'movement_schedule': A list of positions through which the agent is scheduled to move.
            - 'trajectory': A list of positions representing the agent's trajectory.
    """
    def __init__(self, positions: list[tuple[float, float]] = None):
        """
        Initializes the MovementSchedulerCallback with an optional list of positions for the initial movement schedule.

        Args:
            positions (list[tuple[float, float]], optional): An optional list of positions for the initial movement schedule. Defaults to None.
        """
        super().__init__()
        self.positions = positions if positions is not None else list()

    def set_cache(self, cache: Any):
        """
        Sets the cache for the callback and initializes required keys for managing the movement schedule.

        The cache is initialized with the movement schedule ('movement_schedule') and a copy of the schedule
        ('trajectory') for potential use in trajectory analysis or visualization.

        Args:
            cache (Any): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)
        self.cache['movement_schedule'] = self.positions
        self.cache['trajectory'] = deepcopy(self.positions)
        self.requires = [
            'position',
            'move_target',
            'movement_schedule',
            'trajectory'
        ]

    def on_step_end(self, step: int):
        """
        Updates the agent's movement target at the end of each simulation step, based on the movement schedule.

        If the agent does not currently have a movement target ('move_target' is None) and there are remaining positions
        in the movement schedule ('movement_schedule'), the next position is popped from the schedule and set as the new
        movement target.

        Args:
            step (int): The current step of the simulation.
        """
        if len(self.cache['movement_schedule']):
            if self.cache['move_target'] is None:
                self.cache['move_target'] = self.cache['movement_schedule'].pop(0)


class TrajectoryCallback(BaseCallback):
    """
    A callback class designed to manage the trajectory of an agent towards a target position using a TrajectoryManager.

    This callback integrates with a TrajectoryManager to calculate and update the agent's trajectory towards a target
    position. It ensures that the agent follows a smooth path calculated by the TrajectoryManager, based on the agent's
    current position, target position, and direction.

    Attributes:
        trajectory (TrajectoryManager): An instance of TrajectoryManager to manage trajectory calculations.

    Args:
        trajectory_manager (TrajectoryManager): An instance of TrajectoryManager.

    Methods:
        set_cache(cache: Any):
            Sets the cache for the callback and initializes required keys for managing the trajectory and movement schedule.

        on_step_begin(step: int):
            Updates the agent's trajectory and movement schedule at the beginning of each simulation step, based on the current target.

    Notes:
        This callback requires the following keys in the cache:
            - 'position': The current position of the agent.
            - 'move_target': The target position for movement.
            - 'direction': The current direction of the agent in radians.
            - 'movement_schedule': A list of positions through which the agent is scheduled to move.
            - 'trajectory': A list of positions representing the agent's trajectory.
    """
    def __init__(self, trajectory_manager: TrajectoryManager):
        """
        Initializes the TrajectoryCallback with a TrajectoryManager instance.

        Args:
            trajectory_manager (TrajectoryManager): An instance of TrajectoryManager.
        """
        super().__init__()
        self.trajectory = trajectory_manager

    def set_cache(self, cache: Any):
        """
        Sets the cache for the callback and initializes required keys for managing the trajectory and movement schedule.

        Ensures that 'movement_schedule' and 'trajectory' keys are present in the cache, initializing them as empty lists
        if they are not already present.

        Args:
            cache (Any): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)

        if 'trajectory' not in self.cache:
            self.cache['trajectory'] = list()

        self.requires = [
            'position',
            'move_target',
            'direction',
            'movement_schedule',
            'trajectory'
        ]

    def on_step_begin(self, step: int):
        """
        Updates the agent's trajectory and movement schedule at the beginning of each simulation step, based on the current target.

        If the agent has a move target, this method calculates a new trajectory towards that target using the TrajectoryManager.
        The calculated trajectory is then used to update the 'movement_schedule' for the agent, ensuring it follows the calculated
        path. The first position in the updated movement schedule is immediately set as the new 'move_target' for the agent.

        Args:
            step (int): The current step of the simulation.
        """
        if self.cache['move_target'] is not None:
            if self.cache['move_target'] not in self.cache['trajectory']:
                xy = self.trajectory(self.cache['position'], self.cache['move_target'], self.cache['direction'])
                self.cache['trajectory'] = [tuple(item) for item in xy.tolist()]
                self.cache['movement_schedule'] = deepcopy(self.cache['trajectory'])
                self.cache['move_target'] = self.cache['movement_schedule'].pop(0)

