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
        cache['walls_fov'] = [None for _ in range(len(cache['env'].walls))]
        cache['objects_fov'] = [None for _ in range(len(cache['env'].objects))]
        self.requires = ['movement_params', 'walls_fov', 'objects_fov', 'env']
        super().set_cache(cache)

    def on_step_begin(self, step: int):
        """
        Updates the agent's field of view at the beginning of each simulation step based on the current position and direction.

        Args:
            step (int): The current step of the simulation.
        """
        # make that to be in-place
        walls_fov, objects_fov = self.fov(self.cache['movement_params'].position, self.cache['movement_params'].direction)
        for i, wall in enumerate(walls_fov):
            self.cache['walls_fov'][i] = wall
        for i, obj in enumerate(objects_fov):
            self.cache['objects_fov'][i] = obj
        # self.cache['walls_fov'], self.cache['objects_fov'] = self.fov(self.cache['movement_params'].position, self.cache['movement_params'].direction)


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
        cache['walls_ego'] = [None for _ in cache['env'].walls]
        cache['objects_ego'] = [None for _ in cache['env'].objects]
        self.requires = ['walls_ego', 'objects_ego', 'movement_params', 'env']
        super().set_cache(cache)

    def on_step_begin(self, step: int):
        """
        Updates the agent's ego-centric representation at the beginning of each simulation step based on the current position and direction.

        Args:
            step (int): The current step of the simulation.
        """
        if self.cache['movement_params'].position is not None and self.cache['movement_params'].direction is not None:
            walls_ego, objects_ego = self.ego(self.cache['movement_params'].position, self.cache['movement_params'].direction)
            for i, wall in enumerate(walls_ego):
                self.cache['walls_ego'][i] = wall
            for i, obj in enumerate(objects_ego):
                self.cache['objects_ego'][i] = obj
            # self.cache['walls_ego'], self.cache['objects_ego'] = self.ego(self.cache['movement_params'].position, self.cache['movement_params'].direction)


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
        cache['walls_ego_segments'] = [None for _ in cache['walls_ego']]
        cache['objects_ego_segments'] = [None for _ in cache['objects_ego']]
        self.requires = ['walls_ego', 'objects_ego', 'walls_ego_segments', 'objects_ego_segments']
        super().set_cache(cache)

    def on_step_begin(self, step: int):
        """
        Segments the ego-centric representations of walls and objects at the beginning of each simulation step.

        For each list of points representing the ego-centric perception of walls and objects, this method segments them into
        discrete linear segments. The results are stored in the cache under 'walls_ego_segments' and 'objects_ego_segments'.

        Args:
            step (int): The current step of the simulation.
        """
        # if self.cache['walls_ego'] is not None:
        if all([wall is not None for wall in self.cache['walls_ego']]):
            # self.cache['walls_ego_segments'] = list()

            for i, points_ego in enumerate(self.cache['walls_ego']):
                if not points_ego.size:
                    # self.cache['walls_ego_segments'].append(points_ego)
                    self.cache['walls_ego_segments'][i] = points_ego
                else:
                    # self.cache['walls_ego_segments'].append(points2segments(points_ego))
                    self.cache['walls_ego_segments'][i] = points2segments(points_ego)

        # if self.cache['objects_ego'] is not None:
        if all([obj is not None for obj in self.cache['objects_ego']]):
            # self.cache['objects_ego_segments'] = list()

            for i, points_ego in enumerate(self.cache['objects_ego']):
                if not points_ego.size:
                    # self.cache['objects_ego_segments'].append(points_ego)
                    self.cache['objects_ego_segments'][i] = points_ego
                else:
                    # self.cache['objects_ego_segments'].append(points2segments(points_ego))
                    self.cache['objects_ego_segments'][i] = points2segments(points_ego)


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
        cache['walls_pw'] = [None for _ in cache['walls_ego_segments']]
        cache['objects_pw'] = [None for _ in cache['objects_ego_segments']]
        self.requires = ['walls_ego_segments', 'objects_ego_segments', 'walls_pw', 'objects_pw', 'tc_gen']
        super().set_cache(cache)

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
            # self.cache['walls_pw'] = self.cache['tc_gen'].get_grid_activity(
            #     np.concatenate(
            #         [segments for segments in self.cache['walls_ego_segments'] if segments.size]
            #     )
            # )
            # self.cache['walls_pw'] = [
            #     self.cache['tc_gen'].get_grid_activity(segments)
            #     for segments in self.cache['walls_ego_segments']
            # ]
            for i, segments in enumerate(self.cache['walls_ego_segments']):
                self.cache['walls_pw'][i] = self.cache['tc_gen'].get_grid_activity(segments)
        else:
            # self.cache['walls_pw'] = [np.zeros_like(wall) for wall in self.cache['walls_pw']]
            for i, wall in enumerate(self.cache['walls_pw']):
                self.cache['walls_pw'][i] = np.zeros_like(wall)

        if len(self.cache['objects_ego_segments']) and any([segments.size for segments in self.cache['objects_ego_segments']]):
            # self.cache['objects_pw'] = self.cache['tc_gen'].get_grid_activity(
            #     np.concatenate(
            #         [segments for segments in self.cache['objects_ego_segments'] if segments.size]
            #     )
            # )
            # self.cache['objects_pw'] = [
            #     self.cache['tc_gen'].get_grid_activity(segments)
            #     for segments in self.cache['objects_ego_segments']
            # ]
            for i, segments in enumerate(self.cache['objects_ego_segments']):
                self.cache['objects_pw'][i] = self.cache['tc_gen'].get_grid_activity(segments)
        else:
            # self.cache['objects_pw'] = np.zeros_like(self.cache['objects_pw'])
            # self.cache['objects_pw'] = [np.zeros_like(obj) for obj in self.cache['objects_pw']]
            for i, obj in enumerate(self.cache['objects_pw']):
                self.cache['objects_pw'][i] = np.zeros_like(obj)
