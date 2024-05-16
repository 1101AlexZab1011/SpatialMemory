from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from bbtoolkit.utils.attention import AbstractAttention
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.structures import DotDict


@dataclass
class AttentionParams(DotDict):
    """
    A dataclass that stores the parameters related to the attention mechanism.

    Attributes:
        attend_to (int): The index of the object currently being attended to.
        attention_priority (np.ndarray): An array indicating the priority of each object for receiving attention.
        attention_step (int): The current step within the attention cycle.
        attention_cycle (int): The total number of steps in one complete attention cycle.
    """
    attend_to: int = field(default_factory=lambda: None)
    attention_priority: np.ndarray = field(default_factory=lambda: None)
    attention_step: int = field(default_factory=lambda: None)
    attention_cycle: int = field(default_factory=lambda: None)

class AttentionCallback(BaseCallback):
    """
    A callback class that integrates an attention mechanism into a simulation or model by updating the agent's
    attention at the beginning of each simulation step.

    Attributes:
        attn_manager (AbstractAttention): An instance of an AbstractAttention or its subclass that manages the
                                          attention mechanism.

    Methods:
        set_cache(cache: Mapping):
            Sets the cache for the callback and initializes required keys for the field of view.
        on_step_begin(step: int):
            Updates the agent's attention at the beginning of each simulation step based on the current position
            and direction.
    """
    def __init__(self, attn_manager: AbstractAttention):
        """
        Initializes the AttentionCallback with the specified attention manager.

        Args:
            attn_manager (AbstractAttention): An instance of an AbstractAttention or its subclass that manages the
                                              attention mechanism.
        """

        super().__init__()
        self.attn_manager = attn_manager

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache for the callback and initializes required keys for the field of view.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
            on_repeat (str): A flag indicating how to handle repeated keys in the cache. Defaults to 'raise'.
        """
        cache['attention_params']= AttentionParams(attention_cycle=self.attn_manager.cycle)
        super().set_cache(cache, on_repeat)
        self.requires = ['objects_ego', 'attention_params']

    def on_step_begin(self, step: int):
        """
        Updates the agent's attention at the beginning of each simulation step based on the current position and direction.

        Args:
            step (int): The current step of the simulation.
        """
        self.attention_params['attend_to'] = self.attn_manager(self.cache['objects_ego'], return_index=True)
        self.attention_params['attention_priority'] = self.attn_manager.attention_priority
        self.attention_params['attention_step'] = self.attn_manager.timer