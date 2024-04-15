from typing import Mapping

from bbtoolkit.dynamics.attention import AbstractAttention
from bbtoolkit.dynamics.callbacks import BaseCallback


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

    def set_cache(self, cache: Mapping):
        """
        Sets the cache for the callback and initializes required keys for the field of view.

        Args:
            cache (Mapping): A mapping object to be used as the cache for the callback.
        """
        cache['attention_params']= dict(attend_to=None, attention_priority=None)
        super().set_cache(cache)
        self.requires = ['objects_ego', 'attention_params']

    def on_step_begin(self, step: int):
        """
        Updates the agent's attention at the beginning of each simulation step based on the current position and direction.

        Args:
            step (int): The current step of the simulation.
        """
        self.attention_params['attend_to'] = self.attn_manager(self.cache['objects_ego'], return_index=True)
        self.attention_params['attention_priority'] = self.attn_manager.attention_priority