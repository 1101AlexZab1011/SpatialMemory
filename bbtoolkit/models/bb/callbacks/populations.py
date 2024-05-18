import logging
import math
from typing import Mapping

import numpy as np
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.models.bb.neural_generators import GCMap
from bbtoolkit.models.bb.utils import get_pr_cue
from bbtoolkit.structures.tensorgroups import NamedTensor


class HDCallback(BaseCallback):
    """
    A callback designed for handling head direction (HD) cues and updating HD cell activities in an agent-based learning simulation.

    This callback integrates external HD cues, manages HD cell activities based on movement and cognitive states, and updates the agent's perceived direction.

    Attributes:
        init_timesteps (int): The number of initial timesteps during which HD cues are applied.
        hd_cue_scale (float): The scale factor for HD cues.
        no_cue_reset_modes (tuple[str, ...]): Modes in which HD cues are not reset.
        total_steps (int): Counter for the steps during which HD cues are active.
        mode (str): The current mode of the simulation.
    """
    def __init__(
        self,
        init_timesteps: int = 30,
        hd_cue_scale: float = 60,
        no_cue_reset_modes: tuple[str, ...] = ('recall', 'top-down')
    ):
        """
        Initializes the HDCallback instance with specified parameters for HD cue initialization and scaling.

        Args:
            init_timesteps (int): The number of initial timesteps during which HD cues are applied.
            hd_cue_scale (float): The scale factor for HD cues.
            no_cue_reset_modes (tuple[str, ...]): Modes in which HD cues are not reset.
        """
        super().__init__()
        self.init_timesteps = init_timesteps
        self.hd_cue_scale = hd_cue_scale
        self.total_steps = None
        self.mode = None
        self.no_cue_reset_modes = no_cue_reset_modes

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for HD cue management, and initializes HD cues.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `tc_gen`: Transformation circuit generator for HD activity.
            - `movement_params`: Contains the physical movement parameters of the agent, including its current position.
            - `mental_movement_params`: Contains the mental movement parameters of the agent during cognitive navigation tasks.
            - `hd_cue`: HD cue array for external directional cues.
            - `k_ratio`: Ratio of excitation/inhibition for HD cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'dynamics_params',  # Dynamic parameters including dt, mode, and step.
            'tc_gen',  # Transformation circuit generator for HD activity.
            'movement_params',  # Movement parameters including position and direction.
            'mental_movement_params',  # Mental movement parameters for cognitive navigation.
            'hd_cue',  # HD cue array for external directional cues.
            'k_ratio',  # Ratio of excitation/inhibition for HD cells.
            'activity',  # Neural activity levels.
            'connectivity',  # Connectivity matrices between neural populations.
            'weights',  # Synaptic weights between neurons.
            'rates'  # Firing rates of neurons.
        ]

        cache['hd_cue'] = np.zeros(len(cache['weights'].hd.to.hd))
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates HD cues and cell activities at the beginning of each simulation step, based on the agent's movement and cognitive state.

        Args:
            step (int): The current step number.
        """
        if self.mode != self.dynamics_params['mode']:
            if self.dynamics_params['mode'] not in self.no_cue_reset_modes:
                self.total_steps = self.init_timesteps

            self.mode = self.dynamics_params['mode']

        if self.total_steps == self.init_timesteps:
            logging.debug('HD CUE INITIATED')


        if self.total_steps != 0:
            self.total_steps -= 1
            self.hd_cue += self.hd_cue_scale*self.tc_gen.get_hd_activity(np.array([self.movement_params.direction]))
        else:
            if not np.all(self.hd_cue == 0):
                logging.debug('HD CUE REMOVED')
                self.hd_cue *= 0

        rot_weights = None
        match self.dynamics_params['mode']:
            case 'bottom-up':
                params = self.movement_params
                target = self.movement_params.move_target if self.movement_params.move_target is not None else self.movement_params.rotate_target
                position = self.movement_params.position
                direction = self.movement_params.direction
            case 'top-down':
                params = self.mental_movement_params
                target = self.mental_movement_params.move_target if self.mental_movement_params.move_target is not None else self.mental_movement_params.rotate_target
                position = self.mental_movement_params.position
                direction = self.mental_movement_params.direction
            case 'recall':
                params, target, position, direction = None, None, None, None

        if target is not None:
            angle_to_target = math.atan2(
                target[1] - position[1],
                target[0] - position[0]
            ) % (2*np.pi)

            diff = angle_to_target - direction
            if diff > np.pi:
                diff -= 2*np.pi
            elif diff < -np.pi:
                diff += 2*np.pi

            if diff < 0:
                rot_weights = self.weights.rot.to.rot.T
            elif diff > 0:
                rot_weights = self.weights.rot.to.rot
            elif np.isclose(angle_to_target, direction):
                rot_weights = None

        if rot_weights is None:
            rot_weights = np.zeros_like(self.weights.rot.to.rot)

        self.k_ratio.hd = -self.activity.hd +\
            (self.connectivity.hd.to.hd['phi']*self.weights.hd.to.hd@self.rates.hd) +\
            self.hd_cue[:, np.newaxis] +\
            (self.connectivity.rot.to.rot['phi']*rot_weights@self.rates.hd)

        if self.dynamics_params['mode'] in ('recall', 'top-down'):
            self.k_ratio.hd += self.connectivity.opr.to.hd['phi'] * self.weights.opr.to.hd@self.rates.opr

        self.activity.hd += self.dt/self.connectivity.hd.to.hd['tau']*self.k_ratio.hd
        self.rates.hd = 1/(1 + np.exp(-2*self.connectivity.hd.to.hd['beta']*(self.activity.hd - self.connectivity.hd.to.hd['alpha'])))

        # HD estimation
        if self.total_steps == 0 and self.dynamics_params['mode'] != 'recall':
            popmax = np.where(self.rates.hd == np.max(self.rates.hd))[0][0]
            hd_estim = popmax*2*np.pi/(len(self.rates.hd) - 1) % (2*np.pi)

            params.direction = hd_estim


class GCRateCallback(BaseCallback):
    """
    A callback designed to update grid cell (GC) activation rates based on the agent's current or mental position in an agent-based learning simulation.

    This callback uses the agent's physical position in bottom-up mode and the mental position in top-down or recall modes to determine the corresponding grid cell activations.

    Attributes:
        gc_map (GCMap): An instance of GCMap containing the firing rates of grid cells across different locations.
    """
    def __init__(self, gc_map: GCMap):
        """
        Initializes the GCRateCallback instance with a specified GCMap.

        Args:
            gc_map (GCMap): An instance of GCMap to be used for updating grid cell activation rates.
        """
        super().__init__()
        self.gc_map = gc_map.fr

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating GC rates, and prepares for rate updates.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `movement_params`: Contains the physical movement parameters of the agent, including its current position.
            - `mental_movement_params`: Contains the mental movement parameters of the agent during cognitive navigation tasks.
            - `rates`: A structure to store the updated grid cell activation rates.
            - `grid2cart`: A mapping function or structure to convert grid indices to Cartesian coordinates.
        """
        self.requires = [
            'dynamics_params',
            'movement_params',
            'mental_movement_params',
            'rates',
            'grid2cart'
        ]
        super().set_cache(cache, on_repeat)

    def get_grid_location(self, x: float, y: float) -> tuple[int, int]:
        """
        Converts Cartesian coordinates to grid indices.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple[int, int]: The corresponding grid indices.
        """
        return self.cache.grid2cart(x, y)

    def on_step_begin(self, step: int):
        """
        Updates the grid cell activation rates at the beginning of each simulation step, based on the agent's current or mental position.

        Args:
            step (int): The current step number.
        """
        match self.dynamics_params.mode:
            case 'bottom-up':
                if self.movement_params.position is not None:
                    self.rates.gc = np.reshape(self.gc_map[*self.get_grid_location(*self.movement_params.position)], (-1, 1))
            case 'top-down' | 'recall':
                if self.mental_movement_params.position is not None:
                    self.rates.gc = np.reshape(self.gc_map[*self.get_grid_location(*self.mental_movement_params.position)], (-1, 1))


class PCCallback(BaseCallback):
    """
    A callback designed to update place cell (PC) dynamics in an agent-based learning simulation.

    This callback computes the activity and rates of place cells based on various inputs and connectivity parameters, adjusting for intrinsic competition among cells.

    Attributes:
        i_comp (float): The initial value for inhibitory compencation.
        i_comp_scale (float): The addictive scaling factor for inhibitory compencation adjustment.
    """
    def __init__(self, i_comp: float = 0, i_comp_scale: float = 15):
        """
        Initializes the PCCallback instance with specified parameters for intrinsic competition.

        Args:
            i_comp (float): The initial value for inhibitory compencation.
            i_comp_scale (float): The scaling factor for inhibitory compencation adjustment.
        """
        super().__init__()
        self.i_comp = i_comp
        self.i_comp_scale = i_comp_scale

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, adds a tensor for intrinsic competition, and specifies the required cache keys for PC dynamics update.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for PC cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
            - `grid2cart`: A mapping function or structure to convert grid indices to Cartesian coordinates.
        """
        cache['rates'].add_tensor(NamedTensor('i_comp', np.array([self.i_comp]).astype(float)))
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates',
            'grid2cart'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the activity and rates of place cells at the beginning of each simulation step, based on the current mode of operation and intrinsic competition among cells.

        Args:
            step (int): The current step number.
        """
        self.cache['k_ratio'].h = (
            - self.activity.h
            + self.connectivity.h.to.h['phi']*self.weights.h.to.h@self.rates.h
            + self.connectivity.pr.to.h['phi']*self.weights.pr.to.h@self.rates.pr
            + self.connectivity.ovc.to.h['phi']*self.weights.ovc.to.h@self.rates.ovc
            + self.rates.i_comp
        )
        match self.dynamics_params['mode']:
            case 'bottom-up':
                self.cache['k_ratio'].h += self.connectivity.bvc.to.h['phi']*self.weights.bvc.to.h@self.rates.bvc
            case 'recall':
                self.cache['k_ratio'].h += self.connectivity.opr.to.h['phi']*self.weights.opr.to.h@self.rates.opr
            case 'top-down':
                self.cache['k_ratio'].h += self.connectivity.opr.to.h['phi']*self.weights.opr.to.h@self.rates.opr +\
                    self.connectivity.gc.to.h['phi']*self.weights.gc.to.h@self.rates.gc

        self.activity.h += self.dt/self.connectivity.h.to.h['tau']*self.cache['k_ratio'].h
        self.rates.h = 1/(1 + np.exp(-2*self.connectivity.h.to.h['beta']*(self.activity.h - self.connectivity.h.to.h['alpha'])))
        # FIXME: What is the 15 in the equation?
        self.rates.i_comp += self.dt/self.connectivity.ic.to.ic['tau']*(self.i_comp_scale - np.sum(self.rates.h))


class BVCCallback(BaseCallback):
    """
    A callback designed to update the Border Vector Cell (BVC) activity and rates based on the current dynamics of the simulation.

    This callback calculates the BVC activity and rates by considering inputs from various sources, including other BVCs, Object Vector Cells (OVCs), Place Cells (PCs), and possibly others depending on the simulation mode.

    Attributes:
        Requires various parameters from the cache to compute the BVC activity and rates, including dynamics parameters, connectivity, and weights.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating BVC activity and rates, and prepares for computation.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for BVC cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the BVC activity and rates at the beginning of each simulation step, based on the current dynamics and mode of operation.

        Args:
            step (int): The current step number.
        """
        self.k_ratio.bvc = (
            - self.activity.bvc
            + self.connectivity.bvc.to.bvc['phi'] + self.weights.bvc.to.bvc @ self.rates.bvc
            + self.connectivity.ovc.to.bvc['phi']*self.weights.ovc.to.bvc @ self.rates.ovc
            + self.connectivity.pr.to.bvc['phi']*self.weights.pr.to.bvc @ self.rates.pr
        )

        match self.dynamics_params['mode']:
            case 'bottom-up':
                self.k_ratio.bvc += self.connectivity.tr.to.bvc['phi']*self.weights.tr.to.bvc @ np.sum(self.rates.tr, axis=0)
            case 'recall' | 'top-down':
                self.k_ratio.bvc += self.connectivity.h.to.bvc['phi']*self.weights.h.to.bvc @ self.rates.h

        self.activity.bvc += self.dt/self.connectivity.bvc.to.bvc['tau']*self.k_ratio.bvc

        self.rates.bvc = 1/(1 + np.exp(-2*self.connectivity.bvc.to.bvc['beta']*(self.activity.bvc - self.connectivity.bvc.to.bvc['alpha'])))


class OVCCallback(BaseCallback):
    """
    A callback designed to update the Object Vector Cells (OVC) based on various inputs and connectivity in an agent-based learning simulation.

    This callback computes the activity and rates of OVCs by integrating inputs from different sources according to the current mode of the simulation.

    Attributes:
        Requires various parameters from the cache to compute the updates for OVCs, including dynamics parameters, connectivity, weights, and rates from other cell types.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating OVCs, and prepares for computation.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for OVC cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the activity and rates of OVCs at the beginning of each simulation step, based on the current dynamics parameters and inputs from various sources.

        Args:
            step (int): The current step number.
        """
        self.k_ratio.ovc = (
            - self.activity.ovc
            + self.connectivity.ovc.to.ovc['phi']*self.weights.ovc.to.ovc @ self.rates.ovc
            + self.connectivity.bvc.to.ovc['phi']*self.weights.bvc.to.ovc @ self.rates.bvc
        )

        match self.dynamics_params['mode']:
            case 'bottom-up':
                self.k_ratio.ovc += self.connectivity.tr.to.ovc['phi']*self.weights.tr.to.bvc @ np.sum(self.rates.otr, axis=0)
            case 'recall' | 'top-down':
                self.k_ratio.ovc += (
                    self.connectivity.opr.to.ovc['phi']*self.weights.opr.to.ovc @ self.rates.opr
                    + self.connectivity.h.to.ovc['phi']*self.weights.h.to.ovc @ self.rates.h
                )

        self.activity.ovc += self.dt/self.connectivity.ovc.to.ovc['tau']*self.k_ratio.ovc

        self.rates.ovc = 1/(1 + np.exp(-2*self.connectivity.ovc.to.ovc['beta']*(self.activity.ovc - self.connectivity.ovc.to.ovc['alpha'])))


class PRCallback(BaseCallback):
    """
    A callback designed to update perirhinal cortex (PR) related parameters and activities in an agent-based learning simulation.

    This callback computes the PR cue from environmental inputs, updates the k-ratio for PR neurons based on various inputs and connectivity, and finally updates the PR neuron rates.

    Attributes:
        pr_cue_scale (float): A scaling factor for the PR cue to adjust its influence on the PR neuron activities.
    """
    def __init__(self, pr_cue_scale: float = 50):
        """
        Initializes the PRCallback instance with a specified PR cue scale.

        Args:
            pr_cue_scale (float): The scaling factor for the PR cue.
        """
        super().__init__()
        self.pr_cue_scale = pr_cue_scale

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating PR parameters, and prepares for PR cue computation.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `env`: The simulation environment containing walls and their textures.
            - `walls_fov`: A list of numpy arrays, each representing the points of a wall that are within the field of view.
            - `k_ratio`: Ratio of excitation/inhibition for PR cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'dynamics_params',
            'env',
            'walls_fov',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the PR cue, k-ratio, activity, and rates at the beginning of each simulation step, based on the current environmental inputs and connectivity.

        Args:
            step (int): The current step number.
        """
        pr_cue = get_pr_cue(self.env, self.walls_fov)[:, np.newaxis]

        self.pr_cue = self.pr_cue_scale*pr_cue/np.max(pr_cue)

        self.k_ratio.pr = -self.activity.pr +\
            self.connectivity.pr.to.pr['phi']*self.weights.pr.to.pr @ self.rates.pr +\
            self.connectivity.bvc.to.pr['phi']*self.weights.bvc.to.pr @ self.rates.bvc +\
            self.pr_cue

        if self.dynamics_params['mode'] == 'recall' or self.dynamics_params['mode'] == 'top-down':
            self.k_ratio.pr += self.connectivity.h.to.pr['phi']*self.weights.h.to.pr @ self.rates.h

        self.activity.pr += self.dt/self.connectivity.pr.to.pr['tau']*self.k_ratio.pr

        self.rates.pr = 1/(1 + np.exp(-2*self.connectivity.pr.to.pr['beta']*(self.activity.pr - self.connectivity.pr.to.pr['alpha'])))


class oPRCallback(BaseCallback):
    """
    A callback designed to update the oPR (object perirhinal) neuron activations based on environmental cues and internal states in an agent-based learning simulation.

    This callback integrates various inputs, including bottom-up sensory information, recall cues, and attentional focus, to update the oPR neuron activations, which are crucial for object recognition and memory recall processes.

    Attributes:
        opr_cue_scale (float): The scaling factor for the cue signal to the oPR neurons, defaulting to 200.
    """
    def __init__(self, opr_cue_scale: float = 200):
        """
        Initializes the oPRCallback instance with a specified cue scale.

        Args:
            opr_cue_scale (float, optional): The scaling factor for the cue signal to the oPR neurons. Defaults to 200.
        """
        super().__init__()
        self.opr_cue = None
        self.opr_cue_scale = opr_cue_scale

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating oPR neuron activations, and initializes the oPR cue array.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `env`: The simulation environment containing objects and their properties.
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `encoding_params`: Parameters for encoding objects in the environment.
            - `attention_params`: Parameters for managing attentional focus on objects.
            - `k_ratio`: Ratio of excitation/inhibition for oPR cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'env',
            'dynamics_params',
            'encoding_params',
            'attention_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.opr_cue = np.zeros((len(self.env.objects), 1))
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the oPR neuron activations at the beginning of each simulation step, based on the current dynamics mode, attention parameters, and encoding parameters.

        Args:
            step (int): The current step number.
        """
        self.opr_cue *= 0

        match self.dynamics_params['mode']:
            case 'bottom-up':
                if self.attention_params['attend_to'] is not None:
                    self.opr_cue[self.attention_params['attend_to']] = 1
            case 'recall':
                self.opr_cue[self.encoding_params['object_to_recall']] = 1

        self.k_ratio.opr = (
                -self.activity.opr +
                self.connectivity.opr.to.opr['phi']*self.weights.opr.to.opr @ self.rates.opr +
                self.connectivity.h.to.opr['phi']*self.weights.h.to.opr @ self.rates.h +
                self.connectivity.ovc.to.opr['phi']*self.weights.ovc.to.opr @ self.rates.ovc +
                self.opr_cue_scale*self.opr_cue
        )

        self.activity.opr  = self.dt/self.connectivity.pr.to.pr['tau']*self.k_ratio.opr

        self.rates.opr = 1/(1 + np.exp(-2*self.connectivity.pr.to.pr['beta']*(self.activity.opr - self.connectivity.pr.to.pr['alpha'])))


class PWCallback(BaseCallback):
    """
    A callback designed for updating the Parietal Window (PW) related parameters in an agent-based learning simulation.

    This callback computes the activity and rates of the PW based on the current dynamics of the simulation, including interactions with environmental cues and internal cognitive processes.

    Attributes:
        b_cue_scale (float): The scaling factor for boundary cues in the environment.
    """
    def __init__(self, b_cue_scale: float = 48):
        """
        Initializes the PWCallback instance with a specified scaling factor for boundary cues.

        Args:
            b_cue_scale (float): The scaling factor for boundary cues in the environment.
        """
        super().__init__()
        self.b_cue_scale = b_cue_scale

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating PW parameters, and prepares for computation.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for PW cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
            - `walls_pw`: The boundary cues in the environment.
        """
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates',
            'walls_pw'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the PW activity and rates at the beginning of each simulation step, based on the current mode of operation and environmental interactions.

        Args:
            step (int): The current step number.
        """
        self.k_ratio.pw = (
            -self.activity.pw
            - self.connectivity.pw.to.pw['inhibitory_phi']
        )

        match self.dynamics_params['mode']:
            case 'bottom-up':
                self.k_ratio.pw += self.b_cue_scale*np.sum(self.walls_pw, axis=0)[:, np.newaxis]
            case 'recall' | 'top-down':
                self.k_ratio.pw += self.connectivity.tr.to.pw['phi']*np.sum(np.transpose(self.weights.tr.to.pw, (2, 0, 1)) @ self.rates.tr, axis=0)

        self.activity.pw += self.dt/self.connectivity.pw.to.pw['tau']*self.k_ratio.pw

        self.rates.pw = 1/(1 + np.exp(-2*self.connectivity.pw.to.pw['beta']*(self.activity.pw - self.connectivity.pw.to.pw['alpha'])))


class oPWCallback(BaseCallback):
    """
    A callback designed to update the oPW (Object Parietal Window) related parameters and activities in an agent-based learning simulation.

    This callback handles the computation of oPW activity based on the current dynamics mode (bottom-up, recall, top-down), attention parameters, and object cues.

    Attributes:
        o_cue_scale (float): The scaling factor for object cues in the oPW computation.
        attn_prev (Any): The previous attention target, used to log changes in attention.
    """
    def __init__(self, o_cue_scale: float = 40):
        """
        Initializes the oPWCallback instance with a specified object cue scale.

        Args:
            o_cue_scale (float, optional): The scaling factor for object cues. Defaults to 40.
        """
        super().__init__()
        self.attn_prev = None
        self.o_cue_scale = o_cue_scale

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for oPW computation, and prepares for the simulation step.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for oPW cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
            - `attention_params`: The attentional focus parameters.
            - `objects_pw`: The object cues in the oPW computation.
        """
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates',
            'attention_params',
            'objects_pw'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the oPW activity and rates at the beginning of each simulation step, based on the current dynamics mode and attention parameters.

        Args:
            step (int): The current step number.
        """
        self.k_ratio.opw = (
            -self.activity.opw
            - np.sum(self.rates.opw) * self.connectivity.opw.to.opw['inhibitory_phi']
        )

        match self.dynamics_params['mode']:
            case 'bottom-up':
                objects_pw_cue = self.o_cue_scale*self.objects_pw[self.attention_params['attend_to']][:, np.newaxis] if self.attention_params['attend_to'] is not None else 0
                self.k_ratio.opw += objects_pw_cue
            case 'recall' | 'top-down':
                self.k_ratio.opw += self.connectivity.tr.to.opw['phi']*np.sum(np.transpose(self.weights.tr.to.pw, (2, 0, 1)) @ self.rates.otr, axis=0)

        self.activity.opw += self.dt/self.connectivity.opw.to.opw['tau']*self.k_ratio.opw

        self.rates.opw = 1/(1 + np.exp(-2*self.connectivity.opw.to.opw['beta']*(self.activity.opw - self.connectivity.opw.to.opw['alpha'])))

        if self.attn_prev != self.attention_params['attend_to']:
            self.attn_prev = self.attention_params['attend_to']
            logging.debug(f'Switch attention to {self.attention_params["attend_to"]}')


class IPRateCallback(BaseCallback):
    """
    A callback to update the inhibitory potential (IP) rates within a neural simulation.

    This callback calculates the IP rates based on the current connectivity parameters and the sum of head direction (HD) cell rates.

    Attributes:
        Requires connectivity information and current rates for calculations.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating IP rates, and prepares for calculations.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `connectivity`: Connectivity matrices between neural populations.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'connectivity',
            'rates'
        ]
        super().set_cache(cache, on_repeat)

    def on_step_begin(self, step: int):
        """
        Updates the IP rates at the beginning of each simulation step based on the connectivity parameters and HD cell rates.

        Args:
            step (int): The current step number.
        """
        self.rates.ip = np.array([1/(1 + np.exp(-2*self.connectivity.ip.to.ip['beta']*(self.connectivity.hd.to.ip['phi']*np.sum(self.rates.hd) - self.connectivity.ip.to.ip['alpha'])))])


class TCCallback(BaseCallback):
    """
    A callback to update the rates and activities of Transformation Circuit (TC) within a neural simulation.

    This callback calculates the TC rates and activities based on the current dynamics parameters, connectivity, weights, and rates from other cell types.

    Attributes:
        Requires various parameters from the cache to manage and update TC rates and activities.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating TC rates and activities, and prepares for calculations.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for TC cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the TC rates and activities at the beginning of each simulation step based on the dynamics parameters, connectivity, weights, and rates from other cell types.

        Args:
            step (int): The current step number.
        """"""
        Updates the TC rates and activities at the beginning of each simulation step based on the dynamics parameters, connectivity, weights, and rates from other cell types.

        Args:
            step (int): The current step number.
        """
        self.k_ratio.tr = (
            -self.activity.tr
            - np.sum(self.rates.tr, axis=0) * self.connectivity.tr.to.tr['inhibitory_phi']
            + self.connectivity.hd.to.tr['phi']*np.transpose(self.weights.hd.to.tr, (2, 0, 1)) @ self.rates.hd
            - self.connectivity.ip.to.tr['phi']*self.rates.ip
        )

        match self.dynamics_params['mode']:
            case 'bottom-up':
                self.k_ratio.tr += self.connectivity.pw.to.tr['phi']*np.transpose(self.weights.pw.to.tr, (2, 0, 1)) @ self.rates.pw
            case 'recall' | 'top-down':
                self.k_ratio.tr += self.connectivity.bvc.to.tr['phi']*self.weights.bvc.to.tr @ self.rates.bvc

        self.activity.tr += self.dt/self.connectivity.tr.to.tr['tau']*self.k_ratio.tr

        self.rates.tr = 1/(1 + np.exp(-2*self.connectivity.tr.to.tr['beta']*(self.activity.tr - self.connectivity.tr.to.tr['alpha'])))


class oTCCallback(BaseCallback):
    """
    A callback designed for updating the transformation circuit (oTC) for objects in an agent-based learning simulation.

    This callback computes the activity and rates of the oTC neurons based on various inputs and connectivity parameters, adjusting for the current dynamics of the simulation.

    Attributes:
        Requires various parameters from the cache to compute the updates for the oTC neurons.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for updating oTC neurons, and prepares for computation.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `dynamics_params`: Contains dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
            - `k_ratio`: Ratio of excitation/inhibition for oTC cells.
            - `activity`: Neural activity levels.
            - `connectivity`: Connectivity matrices between neural populations.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
        """
        self.requires = [
            'dynamics_params',
            'k_ratio',
            'activity',
            'connectivity',
            'weights',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.dt = self.dynamics_params['dt']

    def on_step_begin(self, step: int):
        """
        Updates the oTC neurons' activity and rates at the beginning of each simulation step, based on the current mode of operation and connectivity parameters.

        Args:
            step (int): The current step number.
        """
        self.k_ratio.otr = (
            - self.activity.otr
            - np.sum(self.rates.otr, axis=0) * self.connectivity.otr.to.otr['inhibitory_phi']
            + self.connectivity.hd.to.tr['phi']*np.transpose(self.cache['weights'].hd.to.tr, (2, 0, 1)) @ self.rates.hd
            - self.connectivity.ip.to.otr['phi']*self.rates.ip
        )

        match self.dynamics_params['mode']:
            case 'bottom-up':
                self.k_ratio.otr += self.connectivity.opw.to.tr['phi']*np.transpose(self.cache['weights'].pw.to.tr, (2, 0, 1)) @ self.rates.opw
            case 'recall' | 'top-down':
                self.k_ratio.otr += self.connectivity.bvc.to.tr['phi']*self.weights.bvc.to.tr @ self.rates.ovc


        self.activity.otr += self.dt/self.connectivity.otr.to.otr['tau']*self.k_ratio.otr

        self.rates.otr = 1/(1 + np.exp(-2*self.connectivity.tr.to.tr['beta']*(self.activity.otr - self.connectivity.tr.to.tr['alpha'])))



