import logging
from typing import Literal, Mapping

import numpy as np
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.structures.tensorgroups import DirectedTensor, DirectedTensorGroup, TensorGroup


class ObjectWeightsUpdatingCallback(BaseCallback):
    """
    A callback designed to update the synaptic weights between different objects-related populations of neurons based on their activity levels during the bottom-up processing mode of an agent-based learning simulation.

    This callback dynamically adjusts the connections between neurons to reflect the learning process as the agent interacts with its environment.

    Attributes:
        init_steps (int): The number of initial steps to ignore before starting the weight updating process.
        rate_threshold (float): The threshold for the maximum rate of a neuron population to consider an object as attended.
        population_thresholds (DirectedTensorGroup): Custom thresholds for updating weights between different neuron populations.
    """
    def __init__(
        self,
        init_steps: int = 10,
        rate_threshold: float = .99,
        population_thresholds: DirectedTensorGroup = None
    ):
        """
        Initializes the ObjectWeightsUpdatingCallback with specified initial steps, rate threshold, and population thresholds.

        Args:
            init_steps (int, optional): The number of initial steps to ignore before starting the weight updating process. Defaults to 10.
            rate_threshold (float, optional): The threshold for the maximum rate of a neuron population to consider an object as attended. Defaults to .99.
            population_thresholds (DirectedTensorGroup, optional): Custom thresholds for updating weights between different neuron populations. Defaults to None.
        """
        super().__init__()
        self.init_steps = init_steps
        self.rate_threshold = rate_threshold
        if population_thresholds is None:
            self.population_thresholds = DirectedTensorGroup(
                DirectedTensor(
                    from_='ovc',
                    to='h',
                    weights=dict(threshold=0.05, update_threshold=0.05)
                ),
                DirectedTensor(
                    from_='ovc',
                    to='opr',
                    weights=dict(threshold=0.05, update_threshold=0.05)
                ),
                DirectedTensor(
                    from_='h',
                    to='opr',
                    weights=dict(threshold=None, update_threshold=0.2)
                ),
                DirectedTensor(
                    from_='hd',
                    to='opr',
                    weights=dict(threshold=None, update_threshold=0.2)
                ),
                DirectedTensor(
                    from_='bvc',
                    to='ovc',
                    weights=dict(threshold=0.05, update_threshold=0.07)
                ),
            )

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes encoded objects if not present, and specifies the required cache keys for updating weights.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.

        Notes:
            The `self.requires` attribute specifies the following required parameters:
            - `activity`: Neural activity levels.
            - `weights`: Synaptic weights between neurons.
            - `rates`: Firing rates of neurons.
            - `encoding_params`: Parameters related to encoding objects in the simulation.
            - `attention_params`: Parameters related to the attentional focus of the agent.
            - `dynamics_params`: Dynamic parameters of the simulation, including the current mode (e.g., 'bottom-up', 'top-down', 'recall').
        """
        if 'encoded_objects' not in cache['encoding_params'] or\
            cache['encoding_params']['encoded_objects'] is None:
            cache['encoding_params'].update(dict(
                encoded_objects=DirectedTensorGroup(
                    DirectedTensor(
                        from_='ovc',
                        to='h',
                        weights=np.zeros(len(cache['env'].objects)).astype(bool)
                    ),
                    DirectedTensor(
                        from_='ovc',
                        to='opr',
                        weights=np.zeros(len(cache['env'].objects)).astype(bool)
                    ),
                    DirectedTensor(
                        from_='h',
                        to='opr',
                        weights=np.zeros(len(cache['env'].objects)).astype(bool)
                    ),
                    DirectedTensor(
                        from_='hd',
                        to='opr',
                        weights=np.zeros(len(cache['env'].objects)).astype(bool)
                    ),
                    DirectedTensor(
                        from_='bvc',
                        to='ovc',
                        weights=np.zeros(len(cache['env'].objects)).astype(bool)
                    )
                )
            ))
        self.requires = [
            'activity',
            'weights',
            'rates',
            'encoding_params',
            'attention_params',
            'dynamics_params'
        ]
        super().set_cache(cache, on_repeat)

    def update_weights_and_encoding(
        self,
        from_act: TensorGroup,
        to_act: TensorGroup,
        from_key: str, to_key: str,
        threshold: float = 0.05,
        update_threshold: float = 0.2,
        update_rule: Literal['forward', 'backward', 'bidirectional'] = 'bidirectional'
    ):
        """
        Updates the synaptic weights and encoding status between specified neuron populations based on their activity levels and thresholds.

        Args:
            from_act (TensorGroup): The activity levels of the source neuron population.
            to_act (TensorGroup): The activity levels of the target neuron population.
            from_key (str): The key identifying the source neuron population.
            to_key (str): The key identifying the target neuron population.
            threshold (float, optional): The activity threshold for considering a neuron as active. Defaults to 0.05.
            update_threshold (float, optional): The threshold for updating synaptic weights. Defaults to 0.2.
            update_rule (Literal['forward', 'backward', 'bidirectional'], optional): The direction of the connection between populations which weights are going to be updated. Defaults to 'bidirectional'.
        """
        if not self.encoding_params['encoded_objects'][from_key].to[to_key][self.attention_params['attend_to']]:
            from_act = np.maximum(from_act, 0)
            to_act = np.maximum(to_act, 0)

            if threshold is not None:
                from_act[from_act < threshold] = 0

            act_product = to_act @ from_act.T
            max_act = np.max(act_product)
            if np.isclose(max_act, 0):
                max_act = 1
            act_product /= max_act

            match update_rule:
                case 'bidirectional':
                    significant_act = act_product > update_threshold
                    update = act_product[significant_act]
                    self.weights[from_key].to[to_key][significant_act] += update
                    self.weights[to_key].to[from_key] = self.weights[from_key].to[to_key].T
                case 'forward':
                    significant_act = act_product > update_threshold
                    update = act_product[significant_act]
                    self.weights[from_key].to[to_key][significant_act] += update
                case 'backward':
                    act_product = act_product.T
                    significant_act = act_product > update_threshold
                    update = act_product[significant_act]
                    self.weights[to_key].to[from_key][significant_act] += update

            self.encoding_params['encoded_objects'][from_key].to[to_key][self.attention_params['attend_to']] = True
            logging.debug(f'{from_key.upper()}2{to_key.upper()} FOR OBJECT {self.attention_params["attend_to"]} UPDATED')

    def on_step_end(self, step: int):
        """
        Executes the weight updating logic at the end of each simulation step, based on the current mode of operation and activity levels.

        Args:
            step (int): The current step number.
        """
        if self.dynamics_params.mode == 'bottom-up' and self.dynamics_params['step'] > self.init_steps:
            ovc_rate_max = self.rates.ovc.max()
            if self.attention_params['attend_to'] is not None and\
                ovc_rate_max > self.rate_threshold and\
                self.cache.attention_params.attention_step > self.cache.attention_params.attention_cycle//3:

                self.update_weights_and_encoding(
                    self.activity.ovc, self.activity.h,
                    'ovc', 'h',
                    self.population_thresholds.ovc.to.h['threshold'],
                    self.population_thresholds.ovc.to.h['update_threshold']
                )
                self.update_weights_and_encoding(
                    self.activity.ovc, self.activity.opr,
                    'ovc', 'opr',
                    self.population_thresholds.ovc.to.opr['threshold'],
                    self.population_thresholds.ovc.to.opr['update_threshold']
                )
                self.update_weights_and_encoding(
                    self.activity.h, self.activity.opr,
                    'h', 'opr',
                    self.population_thresholds.h.to.opr['threshold'],
                    self.population_thresholds.h.to.opr['update_threshold']
                )
                self.update_weights_and_encoding(
                    self.activity.hd, self.activity.opr,
                    'hd', 'opr',
                    self.population_thresholds.hd.to.opr['threshold'],
                    self.population_thresholds.hd.to.opr['update_threshold'],
                    update_rule='backward'
                )
                self.update_weights_and_encoding(
                    self.activity.bvc, self.activity.ovc,
                    'bvc', 'ovc',
                    self.population_thresholds.bvc.to.ovc['threshold'],
                    self.population_thresholds.bvc.to.ovc['update_threshold']
                )