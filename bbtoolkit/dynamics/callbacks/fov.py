from typing import Mapping

import numpy as np
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.math.geometry import points2segments
from bbtoolkit.preprocessing.environment.fov import FOVManager
from bbtoolkit.preprocessing.environment.fov.ego import EgoManager


class FOVCallback(BaseCallback):
    """
    A callback class designed to update the field of view (FOV) of an agent within a simulation environment.

    This callback integrates with an FOVManager to calculate and update the agent's field of view, including visible
    walls and objects, based on the agent's current position and direction.

    Attributes:
        fov (FOVManager): An instance of FOVManager to manage calculations related to the agent's field of view.

    Args:
        fov_manager (FOVManager): An instance of FOVManager.

    Methods:
        set_cache(cache: Mapping):
            Sets the cache for the callback and initializes required keys for the field of view.

        on_step_begin(step: int):
            Updates the agent's field of view at the beginning of each simulation step based on the current position and direction.

    Notes:
        This callback requires the following keys in the cache:
            - position: The current position of the agent.
            - direction: The current direction of the agent in radians.
            - walls_fov: A list of points representing the walls visible within the agent's field of view.
            - objects_fov: A list of points representing the objects visible within the agent's field of view.
    """
    def __init__(self, fov_manager: FOVManager):
        """
        Initializes the FOVCallback with an FOVManager instance.
        """
        super().__init__()
        self.fov = fov_manager

    def set_cache(self, cache: Mapping):
        """
        Sets the cache for the callback and initializes required keys for the field of view.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)
        self.cache['walls_fov'] = None
        self.cache['objects_fov'] = None
        self.requires = ['position', 'direction', 'walls_fov', 'objects_fov']

    def on_step_begin(self, step: int):
        """
        Updates the agent's field of view at the beginning of each simulation step based on the current position and direction.

        Args:
            step (int): The current step of the simulation.
        """
        self.cache['walls_fov'], self.cache['objects_fov'] = self.fov(self.cache['position'], self.cache['direction'])


class EgoCallback(BaseCallback):
    """
    A callback class designed to update the ego-centric representation of an agent within a simulation environment.

    This callback integrates with an EgoManager to calculate and update the agent's ego-centric representation, including
    the relative positions of walls and objects, based on the agent's current position and direction.

    Attributes:
        ego (EgoManager): An instance of EgoManager to manage calculations related to the agent's ego-centric representation.

    Args:
        ego_manager (EgoManager): An instance of EgoManager.

    Methods:
        set_cache(cache: Mapping):
            Sets the cache for the callback and initializes required keys for the ego-centric representation.

        on_step_begin(step: int):
            Updates the agent's ego-centric representation at the beginning of each simulation step based on the current position and direction.

    Notes:
        This callback requires the following keys in the cache:
            - position: The current position of the agent.
            - direction: The current direction of the agent in radians.
            - walls_ego: A list of points representing the walls within the agent's ego-centric representation.
            - objects_ego: A list of points representing the objects within the agent's ego-centric representation.
    """
    def __init__(self, ego_manager: EgoManager):
        """
        Initializes the EgoCallback with an EgoManager instance.

        Args:
            ego_manager (EgoManager): An instance of EgoManager.
        """
        super().__init__()
        self.ego = ego_manager

    def set_cache(self, cache: Mapping):
        """
        Sets the cache for the callback and initializes required keys for the ego-centric representation.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)
        self.cache['walls_ego'] = None
        self.cache['objects_ego'] = None
        self.requires = ['walls_ego', 'objects_ego', 'position', 'direction']

    def on_step_begin(self, step: int):
        """
        Updates the agent's ego-centric representation at the beginning of each simulation step based on the current position and direction.

        Args:
            step (int): The current step of the simulation.
        """
        if self.cache['position'] is not None and self.cache['direction'] is not None:
            self.cache['walls_ego'], self.cache['objects_ego'] = self.ego(self.cache['position'], self.cache['direction'])


class EgoSegmentationCallback(BaseCallback):
    """
    A callback class designed to segment the ego-centric representations of walls and objects into discrete segments.

    This callback processes the ego-centric representations of walls and objects, provided as lists of points, and segments
    them into discrete, linear segments. This is useful for further processing or visualization of the agent's perception
    of its environment.

    Methods:
        set_cache(cache: Mapping):
            Sets the cache for the callback and initializes required keys for storing segmented representations.

        on_step_begin(step: int):
            Segments the ego-centric representations of walls and objects at the beginning of each simulation step.

    Notes:
        This callback requires the following keys in the cache:
            - walls_ego: A list of points representing the walls within the agent's ego-centric representation.
            - objects_ego: A list of points representing the objects within the agent's ego-centric representation.
            - walls_ego_segments: A list of segmented representations of the walls within the agent's ego-centric representation.
            - objects_ego_segments: A list of segmented representations of the objects within the agent's ego-centric representation.
    """
    def set_cache(self, cache: Mapping):
        """
        Sets the cache for the callback and initializes required keys for storing segmented representations of walls and objects.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)
        self.cache['walls_ego_segments'] = list()
        self.cache['objects_ego_segments'] = list()
        self.requires = ['walls_ego', 'objects_ego', 'walls_ego_segments', 'objects_ego_segments']

    def on_step_begin(self, step: int):
        """
        Segments the ego-centric representations of walls and objects at the beginning of each simulation step.

        For each list of points representing the ego-centric perception of walls and objects, this method segments them into
        discrete linear segments. The results are stored in the cache under 'walls_ego_segments' and 'objects_ego_segments'.

        Args:
            step (int): The current step of the simulation.
        """
        if self.cache['walls_ego'] is not None:
            self.cache['walls_ego_segments'] = list()

            for points_ego in self.cache['walls_ego']:
                if not points_ego.size:
                    self.cache['walls_ego_segments'].append(points_ego)
                else:
                    self.cache['walls_ego_segments'].append(points2segments(points_ego))

        if self.cache['objects_ego'] is not None:
            self.cache['objects_ego_segments'] = list()
            for points_ego in self.cache['objects_ego']:
                if not points_ego.size:
                    self.cache['objects_ego_segments'].append(points_ego)
                else:
                    self.cache['objects_ego_segments'].append(points2segments(points_ego))


class ParietalWindowCallback(BaseCallback):
    """
    A callback class designed to update the parietal window representation of walls and objects within a simulation environment.

    This callback processes the segmented ego-centric representations of walls and objects, converting them into a parietal
    window representation. This involves transforming the segmented points into grid activity patterns using a provided
    transformation generator (e.g., a place cell or grid cell model). The parietal window representation is useful for
    cognitive and navigational tasks within the simulation.

    Methods:
        set_cache(cache: Mapping):
            Sets the cache for the callback and initializes required keys for storing parietal window representations.

        on_step_begin(step: int):
            Updates the parietal window representations of walls and objects at the beginning of each simulation step.

    Notes:
        This callback requires the following keys in the cache:
            - walls_ego_segments: A list of segmented representations of the walls within the agent's ego-centric representation.
            - objects_ego_segments: A list of segmented representations of the objects within the agent's ego-centric representation.
            - walls_pw: A grid activity pattern representing the parietal window representation of walls.
            - objects_pw: A grid activity pattern representing the parietal window representation of objects.
            - tc_gen: A transformation circuit generator to be used for converting segmented representations into BVCs or OVCs grid activity patterns.
    """
    def set_cache(self, cache: Mapping):
        """
        Sets the cache for the callback and initializes required keys for storing parietal window representations of walls and objects.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
        """
        super().set_cache(cache)
        self.cache['walls_pw'] = None
        self.cache['objects_pw'] = None
        self.requires = ['walls_ego_segments', 'objects_ego_segments', 'walls_pw', 'objects_pw', 'tc_gen']

    def on_step_begin(self, step: int):
        """
        Updates the parietal window representations of walls and objects at the beginning of each simulation step.

        This method transforms the segmented ego-centric representations of walls and objects into grid activity patterns,
        representing the agent's cognitive map of its environment. The transformation is performed using the transformation
        generator specified in the cache under 'tc_gen'.

        Args:
            step (int): The current step of the simulation.
        """
        if len(self.cache['walls_ego_segments']) and any([segments.size for segments in self.cache['walls_ego_segments']]):
            self.cache['walls_pw'] = self.cache['tc_gen'].get_grid_activity(
                np.concatenate(
                    [segments for segments in self.cache['walls_ego_segments'] if segments.size]
                )
            )
        else:
            self.cache['walls_pw'] = np.zeros_like(self.cache['walls_pw'])

        if len(self.cache['objects_ego_segments']) and any([segments.size for segments in self.cache['objects_ego_segments']]):
            self.cache['objects_pw'] = self.cache['tc_gen'].get_grid_activity(
                np.concatenate(
                    [segments for segments in self.cache['objects_ego_segments'] if segments.size]
                )
            )
        else:
            self.cache['objects_pw'] = np.zeros_like(self.cache['objects_pw'])
