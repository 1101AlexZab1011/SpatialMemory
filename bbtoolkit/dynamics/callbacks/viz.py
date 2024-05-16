import math
from typing import Mapping
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.structures import BaseCallbacksManager, BaseCallback as _BaseCallback, CallbacksCollection
from bbtoolkit.utils.viz import show_figure


class ArtistCallback(_BaseCallback):
    """
    A callback class designed for handling plotting-related tasks within a simulation or iterative process.

    This class extends the _BaseCallback class, providing specific methods for plotting and cleaning up plots.
    """
    def on_plot(self):
        """
        Called to execute the plotting logic.
        """
        ...
    def on_clean(self):
        """
        Called to clean up or clear figures before plotting anew.
        """
        ...
    def on_copy(self, **kwargs):
        """
        Called when the callback is copied.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        ...
    def on_load(self, **kwargs):
        """
        Called when the callback is loaded from a serialized state.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        ...


class PlottingCallback(BaseCallbacksManager, BaseCallback):
    """
    A callback class that manages plotting callbacks and integrates with the BaseCallbacksManager.

    This class is responsible for managing artist callbacks, updating plots at a specified rate, and handling figure and grid specifications for plotting.

    This class acts as both a BaseCallback and a BaseCallbacksManager, allowing it to manage a collection of ArtistCallback and execute them in a specific order, while being a callback of a parent BaseCallbacksManager.

    Attributes:
        update_rate (int): The rate at which the plot should be updated.
        fig_kwargs (dict): Keyword arguments for figure creation.
        gc_kwargs (dict): Keyword arguments for grid specification.

    Inherits:
        BaseCallbacksManager: For managing a collection of callbacks.
        BaseCallback: For callback functionality.
    """
    def __init__(self, callbacks: list[ArtistCallback] = None, update_rate: int = 10, fig_kwargs: dict = None, gc_kwargs: dict = None):
        """
        Initializes the PlottingCallback instance.

        Args:
            callbacks (list[ArtistCallback], optional): A list of ArtistCallback instances.
            update_rate (int, optional): The rate at which the plot should be updated. Defaults to 10.
            fig_kwargs (dict, optional): Keyword arguments for figure creation. Defaults to None.
            gc_kwargs (dict, optional): Keyword arguments for grid specification. Defaults to None.
        """
        self.update_rate = update_rate
        fig_kwargs = fig_kwargs or dict()
        gc_kwargs = gc_kwargs or dict()
        self.fig_kwargs = fig_kwargs
        self.gc_kwargs = gc_kwargs
        self.callbacks = CallbacksCollection() if callbacks is None else CallbacksCollection(callbacks)
        BaseCallback.__init__(self)

    @property
    def cache(self):
        """
        Returns the current cache.

        Returns:
            Mapping: The current cache.
        """
        return self._cache

    @cache.setter
    def cache(self, cache: Mapping):
        """
        Sets the cache with the provided mapping.

        Args:
            cache (Mapping): The new cache mapping.
        """
        self._cache = cache

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and initializes figure and grid specifications.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['fig', 'gc']
        if 'fig' not in cache or not isinstance(cache['fig'], plt.Figure):
            cache['fig'] = plt.figure(**self.fig_kwargs)
            cache['gc'] = GridSpec(**self.gc_kwargs, figure=cache['fig'])
        elif 'fig' in cache and isinstance(cache['fig'], plt.Figure):
            cache['fig'].clf()
        else:
            raise ValueError(f'Invalid cache key for fig: {type(cache["fig"])}')

        super().set_cache(cache, on_repeat=on_repeat)
        self.callbacks.execute('set_cache', cache, on_repeat=on_repeat)
        try:
            self.callbacks.validate()
        except TypeError as e:
            raise TypeError(
                f'Error in {self.__class__.__name__}: Failed to validate callbacks due to: {e}\n'
                f'Note: {self.__class__.__name__} acts as both a BaseCallback and a BaseCallbacksManager.\n'
                f'This means that callbacks within {self.__class__.__name__} are nested within the scope of any external callbacks manager utilizing {self.__class__.__name__}.\n'
                'As a result, these nested callbacks have their own separate visibility scope.\n'
                f'If these nested callbacks depend on cache keys available in the external callbacks manager’s cache, they must be positioned before {self.__class__.__name__} in the execution order.'
            )

        cache['fig'].tight_layout()

    def validate_fig(self):
        """
        Validates the existence of the figure and displays it if not already visible.
        """
        if not plt.fignum_exists(self.fig.number):
            show_figure(self.fig)

    def on_step_end(self, step: int):
        """
        Called at the end of a step. Updates the plot based on the update rate.

        Args:
            step (int): The current step number.
        """
        if not step % self.update_rate:
            self.plot()

    def on_simulation_end(self):
        """
        Called at the end of the simulation. Closes the plot.
        """
        plt.close(self.fig)

    def plot(self):
        """
        Validates the figure, cleans up previous plots, executes plotting callbacks, and refreshes the plot.
        """
        self.validate_fig()
        self.callbacks.execute('on_clean')
        self.callbacks.execute('on_plot')
        self.cache['fig'].canvas.draw()
        plt.pause(.00001)

    def on_copy(self, **kwargs):
        """
        Called when the callback is copied.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.callbacks.validate()
        self.callbacks.execute('on_copy', **kwargs)

    def on_load(self, **kwargs):
        """
        Called when the callback is loaded from a serialized state.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.callbacks.validate()
        self.callbacks.execute('on_load', **kwargs)


class AgentPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the agent's position and direction in an agent-based learning simulation.

    This callback visualizes the agent's current position and the direction it is facing, aiding in understanding the agent's state and intended actions.

    Attributes:
        agent_color (str): The color used to plot the agent. Defaults to 'tab:blue'.
    """
    def __init__(self, agent_color: str = 'tab:blue'):
        """
        Initializes the AgentPlotter instance with a specified color for the agent.

        Args:
            agent_color (str, optional): The color for the agent. Defaults to 'tab:blue'.
        """
        super().__init__()
        self.agent_color = agent_color
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and specifies the required cache keys for plotting the agent.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'alo_ax',
            'movement_params'
        ]
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the agent's position and direction, using the specified color for the agent.
        """
        if self.movement_params.position is not None and self.movement_params.direction is not None:
            self.alo_ax.plot(*self.movement_params.position, 'o', color=self.agent_color, zorder=4.5)
            self.alo_ax.arrow(
                *self.movement_params.position,
                0.5 * math.cos(self.movement_params.direction),
                0.5 * math.sin(self.movement_params.direction),
                zorder=4.5
            )



