import logging
import math
from typing import Mapping

from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
import numpy as np
from bbtoolkit.dynamics.callbacks import BaseCallback
from bbtoolkit.dynamics.callbacks.viz import ArtistCallback
from bbtoolkit.utils.math.geometry import calculate_polar_distance
from bbtoolkit.utils.viz import plot_arrow, plot_polygon
from bbtoolkit.utils.viz.colors import adjust_color_brightness, get_most_visible_color
from shapely import Point
import matplotlib.colors as mcolors


class AloEnvPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the environment in an agent-based learning simulation.

    This callback handles the visualization of the environment, including walls, objects, and the agent's field of view (FOV).

    Attributes:
        attn_color (str): The color for the agent's attention.
        min_xy (tuple): Minimum x and y coordinates for the plot boundaries.
        max_xy (tuple): Maximum x and y coordinates for the plot boundaries.
    """
    def __init__(self, attn_color: str = 'tab:red'):
        """
        Initializes the AloEnvPlotter instance.

        Args:
            attn_color (str): The color for the agent's attention. Defaults to 'tab:red'.
        """
        super().__init__()
        self.attn_color = attn_color
        self.min_xy = None
        self.max_xy = None

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and initializes the plot axis for the environment.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['alo_ax'] = cache.fig.add_subplot(cache.gc[:4, :4])
        self.requires = [
            'env',
            'walls_fov',
            'objects_fov',
            'alo_ax',
            'attention_params'
        ]
        super().set_cache(cache, on_repeat)
        coords_x, coords_y = self.env.visible_area.boundary.coords.xy
        min_train_x, max_train_x, min_train_y, max_train_y = min(coords_x), max(coords_x), min(coords_y), max(coords_y)
        self.min_xy = (min_train_x, min_train_y)
        self.max_xy = (max_train_x, max_train_y)

    def plot_environment(self):
        """
        Plots the environment, including walls and objects.
        """
        for obj in self.env.objects + self.env.walls:
            plot_polygon(obj.polygon, ax=self.alo_ax, alpha=0.5, linewidth=1)

    def plot_fov(self):
        """
        Plots the agent's field of view, showing visible walls and objects.
        """
        if self.walls_fov:
            for wall, poly in zip(self.walls_fov, self.env.walls):
                self.alo_ax.plot(
                    wall[:, 0],
                    wall[:, 1],
                    'o',
                    color=poly.polygon.texture.color,
                    markersize=2,
                    markeredgecolor=adjust_color_brightness(poly.polygon.texture.color)
                )
        if self.objects_fov:
            for i, (obj, poly) in enumerate(zip(self.objects_fov, self.env.objects)):
                if self.attention_params.attend_to is not None and i == self.attention_params.attend_to:
                    self.alo_ax.plot(
                        obj[:, 0],
                        obj[:, 1],
                        'o',
                        color=self.attn_color,
                        markersize=3,
                        markeredgecolor=adjust_color_brightness(self.attn_color)
                    )
                else:
                    self.alo_ax.plot(
                        obj[:, 0],
                        obj[:, 1],
                        'o',
                        color=poly.polygon.texture.color,
                        markersize=2,
                        markeredgecolor=adjust_color_brightness(poly.polygon.texture.color)
                    )

    def on_plot(self):
        """
        Executes the plotting logic, including the environment and the agent's field of view.
        """
        self.plot_environment()
        self.plot_fov()

    def on_clean(self):
        """
        Clears the plot in preparation for the next update, and sets the axis limits based on the environment boundaries.
        """
        self.alo_ax.clear()
        self.alo_ax.set_axis_off()
        self.alo_ax.set_xlim(self.min_xy[0], self.max_xy[0])
        self.alo_ax.set_ylim(self.min_xy[1], self.max_xy[1])


class TargetPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting movement and rotation targets in an agent-based learning simulation.

    This callback visualizes targets for movement and rotation, aiding in understanding the agent's intended actions.

    Attributes:
        move_target_color (str): The color used to plot the movement target. Defaults to 'tab:red'.
        rotate_target_color (str): The color used to plot the rotation target. Defaults to 'tab:green'.
    """
    def __init__(
        self,
        move_target_color: str = 'tab:red',
        rotate_target_color: str = 'tab:green'
    ):
        """
        Initializes the TargetPlotter instance with specified colors for movement and rotation targets.

        Args:
            move_target_color (str, optional): The color for movement targets. Defaults to 'tab:red'.
            rotate_target_color (str, optional): The color for rotation targets. Defaults to 'tab:green'.
        """
        super().__init__()
        self.move_target_color = move_target_color
        self.rotate_target_color = rotate_target_color

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and specifies the required cache keys for plotting targets.

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
        Executes the plotting logic for movement and rotation targets, using specified colors for each.
        """
        if self.movement_params.move_target is not None:
            self.alo_ax.plot(*self.movement_params.move_target, 'x', color=self.move_target_color, zorder=3.5)
        if self.movement_params.rotate_target is not None:
            self.alo_ax.plot(*self.movement_params.rotate_target, 'x', color=self.rotate_target_color, zorder=3.5)


class TrajectoryPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the agent's trajectory and movement targets in an agent-based learning simulation.

    This callback visualizes the path that the agent has taken or will take, along with the final target position, aiding in understanding the agent's movement strategy.

    Attributes:
        traj_color (str): The color used to plot the trajectory. Defaults to 'tab:green'.
        target_color (str): The color used to plot the final target position. Defaults to 'tab:red'.
    """
    def __init__(self, traj_color: str = 'tab:green', target_color: str = 'tab:red'):
        """
        Initializes the TrajectoryPlotter instance with specified colors for the trajectory and target.

        Args:
            traj_color (str, optional): The color for the trajectory. Defaults to 'tab:green'.
            target_color (str, optional): The color for the final target position. Defaults to 'tab:red'.
        """
        super().__init__()
        self.traj_color = traj_color
        self.target_color = target_color

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and specifies the required cache keys for plotting the trajectory and target.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'alo_ax',
            'movement_params',
            'movement_schedule',
            'trajectory'
        ]
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the agent's trajectory and movement target, using specified colors for each.
        """
        if self.movement_params.position is not None and \
            (not len(self.trajectory) or
            not (
                self.movement_params.move_target is not None
                and self.movement_params.move_target not in self.trajectory
            )):
            first_points = [self.movement_params.position, self.movement_params.move_target]\
                if self.movement_params.move_target not in self.movement_schedule\
                and self.movement_params.move_target is not None\
                else [self.movement_params.position]
            all_points = first_points + self.movement_schedule
            if len(self.movement_schedule):
                self.alo_ax.plot(
                    self.movement_schedule[-1][0],
                    self.movement_schedule[-1][1],
                    'X', color=self.target_color,
                    zorder=3.5
                )
            for from_, to in zip(all_points[:-1], all_points[1:]):
                self.alo_ax.plot(*zip(from_, to), '-', color=self.traj_color, alpha=.5, zorder=2.5)


class EgoEnvPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting an ego-centric view of the environment in an agent-based learning simulation.

    This callback visualizes the environment from the agent's perspective, including walls and objects, with special attention to objects of interest.

    Attributes:
        attn_color (str): The color used to highlight objects of interest. Defaults to 'tab:red'.
        min_xy (tuple): Minimum x and y coordinates for the plot boundaries.
        max_xy (tuple): Maximum x and y coordinates for the plot boundaries.
    """
    def __init__(self, attn_color: str = 'tab:red'):
        """
        Initializes the EgoEnvPlotter instance with a specified color for highlighting objects of interest.

        Args:
            attn_color (str, optional): The color for highlighting objects of interest. Defaults to 'tab:red'.
        """
        super().__init__()
        self.attn_color = attn_color
        self.min_xy = None
        self.max_xy = None

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and initializes the plot axis for the ego-centric view.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache.ego_ax = cache.fig.add_subplot(cache.gc[4:8, :4])
        self.requires = [
            'env',
            'walls_ego_segments',
            'objects_ego_segments',
            'ego_ax',
            'attention_params'
        ]
        super().set_cache(cache, on_repeat)
        coords_x, coords_y = self.env.visible_area.boundary.coords.xy
        min_train_x, max_train_x, min_train_y, max_train_y = min(coords_x), max(coords_x), min(coords_y), max(coords_y)
        d = np.sqrt((max_train_x - min_train_x)**2 + (max_train_y - min_train_y)**2)
        middle = (max_train_x + min_train_x)/2, (max_train_y + min_train_y)/2
        self.min_xy = (middle[0] - d, -1)
        self.max_xy = (middle[0] + d, d)

    def plot_ego(self):
        """
        Plots the ego-centric representation of walls and objects.
        """
        _ = plot_arrow(np.pi/2, 0, -.75, ax=self.ego_ax)

        if self.walls_ego_segments:
            for segments, poly in zip(self.walls_ego_segments, self.env.walls):
                for seg in segments:
                    x_start, y_start, x_end, y_end = seg
                    self.ego_ax.plot([x_start, x_end], [y_start, y_end], color=poly.polygon.texture.color, linewidth=1)

        if self.objects_ego_segments:
            for i, (segments, poly) in enumerate(zip(self.objects_ego_segments, self.env.objects)):
                color = self.attn_color if self.attention_params.attend_to is not None and i == self.attention_params['attend_to'] else poly.polygon.texture.color
                for seg in segments:
                    x_start, y_start, x_end, y_end = seg
                    self.ego_ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=1)

    def on_plot(self):
        """
        Executes the plotting logic for the ego-centric view of the environment.
        """
        self.plot_ego()

    def on_clean(self):
        """
        Clears the plot in preparation for the next update, sets the axis ticks and limits based on the ego-centric view boundaries.
        """
        self.ego_ax.clear()
        self.ego_ax.set_xticks([])
        self.ego_ax.set_yticks([])

        for spine in self.ego_ax.spines.values():
            spine.set_edgecolor('tab:grey')

        self.ego_ax.set_xlim(self.min_xy[0], self.max_xy[0])
        self.ego_ax.set_ylim(self.min_xy[1], self.max_xy[1])


class MouseEventCallback(ArtistCallback):
    """
    A specialized ArtistCallback for handling mouse events within the plotting area of an agent-based learning simulation.

    This callback enables interactive setting of movement and rotation targets through mouse clicks on the plot.

    Attributes:
        Requires various parameters from the cache to handle mouse events effectively, including the figure object for connecting events and environmental data for determining click positions relative to objects and walls.
    """
    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping and connects the mouse click event to the on_click method.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'fig',
            'env',
            'dynamics_params',
            'movement_params',
            'mental_movement_params',
            'click_params',
            'alo_ax'
        ]
        super().set_cache(cache, on_repeat)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    @staticmethod
    def point_outside_bounds(x, y, objects):
        """
        Determines if a clicked point is outside the bounds of specified objects.

        Args:
            x (float): The x-coordinate of the clicked point.
            y (float): The y-coordinate of the clicked point.
            objects (list): A list of objects to check against.

        Returns:
            bool or np.array: False if the point is outside all objects, otherwise the indices of objects containing the point.
        """
        point = Point(x, y)
        is_contained = np.array([
            obj.polygon.contains(point)
            for obj in objects
        ])
        if not np.any(is_contained):
            return False
        else:
            return np.where(is_contained)[0]

    def on_click(self, event: MouseEvent):
        """
        Handles mouse click events on the plot for setting movement and rotation targets.

        Args:
            event (MouseEvent): The mouse click event on the plot.
        """

        if event.inaxes is self.alo_ax:

            self.click_params.xy_data = (event.xdata, event.ydata)
            self.click_params.inside_object = self.point_outside_bounds(event.xdata, event.ydata, self.env.objects)
            self.click_params.inside_wall = self.point_outside_bounds(event.xdata, event.ydata, self.env.walls)

            match self.dynamics_params.mode:
                case 'bottom-up':
                    movement_params = self.movement_params
                case 'recall' | 'top-down':
                    movement_params = self.mental_movement_params

            if self.dynamics_params.mode == 'recall' and\
                (
                    (
                        self.click_params['inside_object'] is False
                        and self.click_params['inside_wall'] is False
                    ) or event.button is MouseButton.RIGHT
                ):
                    logging.debug('Switching to top-down mode')
                    self.dynamics_params.mode = 'top-down'

            # Be aware of checking self.click_params['inside_object'], since it can be either false of np.array which may not survive if-else (in the case of np.array([0]))
            if event.button is MouseButton.LEFT and self.click_params['inside_object'] is False and self.click_params['inside_wall'] is False:
                movement_params.move_target = event.xdata, event.ydata
                movement_params.rotate_target = None
            elif event.button is MouseButton.RIGHT:
                movement_params.rotate_target = event.xdata, event.ydata
                movement_params.move_target = None

    def on_copy(self):
        """
        Reconnects the mouse click event handler when the callback is copied.
        """
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_load(self):
        """
        Reconnects the mouse click event handler when the callback is loaded from a serialized state.
        """
        self.on_copy()


class ObjectRecallCallback(ArtistCallback):
    """
    A callback designed to handle object recall interactions in an agent-based learning simulation.

    This callback allows for the initiation and termination of object recall processes through keyboard interactions, enabling the simulation to switch between different modes of operation based on user input.

    Attributes:
        prev_mode (str): The previous mode of the dynamics parameters before initiating recall.
    """
    def __init__(self):
        """
        Initializes the ObjectRecallCallback instance.
        """
        super().__init__()
        self.prev_mode = None

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for object recall, and connects the key press event to the on_press method.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['encoding_params', 'click_params', 'dynamics_params', 'fig']
        super().set_cache(cache, on_repeat)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def validate_encoding(self):
        """
        Validates that the selected object is encoded in the simulation's memory, raising an error if recall is not possible.
        """
        for from_, data_from in self.encoding_params['encoded_objects'].data.items():
                for to, data_to in data_from.items():
                    if not data_to[self.click_params["inside_object"][0]]:
                        raise ValueError(
                            f'Object {self.click_params["inside_object"][0]} is not encoded in {from_}2{to} weights. Recall is not possible.'
                        )

    def on_press(self, event: KeyEvent):
        """
        Handles key press events for initiating and terminating object recall, as well as switching simulation modes.

        Args:
            event (KeyEvent): The key press event.
        """
        if event.key == 'r' and self.click_params['inside_object'] is not False:

            if len(self.click_params['inside_object']) > 1:
                    raise ValueError(
                        f'Several objects are selected: {self.click_params["inside_object"]}. '
                        'Likely, these objects are overlapped. Recall is not possible for multiple objects at the same time'
                    )

            if self.dynamics_params['mode'] != 'recall' or\
                (
                    self.dynamics_params['mode'] == 'recall' and
                    self.click_params['inside_object'][0] != self.encoding_params['object_to_recall']
                ):
                logging.debug(f'Initiate recall for object {self.click_params["inside_object"][0]}')
                self.validate_encoding()
                if self.dynamics_params['mode'] != 'recall':
                    self.prev_mode = self.dynamics_params['mode']
                self.dynamics_params['mode'] = 'recall'
                self.encoding_params['object_to_recall'] = self.click_params['inside_object'][0]
            else:
                logging.debug('Stop recall')
                self.dynamics_params['mode'] = self.prev_mode
                self.encoding_params['object_to_recall'] = None

        if event.key == 'p':
            logging.debug('Switch to bottom-up mode')
            self.dynamics_params['mode'] = 'bottom-up'

            if self.encoding_params['object_to_recall'] is not None:
                self.encoding_params['object_to_recall'] = None

    def on_copy(self):
        """
        Reconnects the key press event handler when the callback is copied.
        """
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def on_load(self):
        """
        Reconnects the key press event handler when the callback is loaded from a serialized state.
        """
        self.on_copy()


class TimerPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the current simulation time on the plot in an agent-based learning simulation.

    This callback visualizes the simulation time, providing a real-time update on the plot as the simulation progresses.

    Attributes:
        coords (tuple): Coordinates on the plot where the simulation time text will be displayed.
    """
    def __init__(self):
        """
        Initializes the TimerPlotter instance.
        """
        super().__init__()
        self.coords = None

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting the time, and calculates the coordinates for displaying the time text.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['alo_ax', 'dynamics_params', 'env']
        super().set_cache(cache, on_repeat)
        x, y = cache.env.visible_area.boundary.xy
        self.coords = min(x), max(y) + 1

    def plot(self):
        """
        Plots the current simulation time on the plot.
        """
        self.alo_ax.text(*self.coords, f'Time: {(self.dynamics_params.step)*self.dynamics_params.dt : .2f} s')

    def on_plot(self):
        """
        Executes the plotting logic for displaying the current simulation time.
        """
        self.plot()


class PWPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting Parietal Window (PW) representations in an agent-based learning simulation.

    This callback visualizes the agent's perception of walls in a polar coordinate system, providing insights into the agent's navigational state and environmental understanding.

    Attributes:
        Requires various parameters from the cache to plot the PW representation, including a dedicated axis for polar plotting and data related to the agent's perception and the environment.
    """
    def __init__(self, cmap: str | mcolors.Colormap = 'coolwarm'):
        """
        Initializes the PWPlotter instance with a specified colormap for the polar plot.

        Args:
            cmap (str | mcolors.Colormap, optional): The colormap for the polar plot. Defaults to 'coolwarm'.
        """
        super().__init__()
        self.cmap = cmap
        self.grid_color = get_most_visible_color(self.cmap)

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the polar plot axis, and calculates necessary parameters for plotting the PW representation.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['pw_ax'] = cache.fig.add_subplot(cache.gc[4:8, 4:8], projection='polar')
        self.requires = [
            'pw_ax',
            'rates',
            'env',
            'walls_pw',
            'tc_gen',
            'theta_bvc',
            'r_bvc'
        ]
        polar_distance = calculate_polar_distance(cache.tc_gen.r_max)
        polar_angle = np.linspace(
            0,
            (cache.tc_gen.n_bvc_theta + 1) * cache.tc_gen.polar_ang_res,
            cache.tc_gen.n_bvc_theta
        )
        polar_distance, polar_angle = np.meshgrid(polar_distance, polar_angle)
        cache['theta_bvc'], cache['r_bvc'] = np.meshgrid(
            np.linspace(0, 2 * np.pi, cache.tc_gen.n_bvc_theta),  # Angular dimension
            np.linspace(0, 1, cache.tc_gen.n_bvc_r)  # Radial dimension, adjust as necessary
        )
        super().set_cache(cache, on_repeat)

    def plot(self):
        """
        Plots the ego-centric representation of walls and objects.
        """
        self.pw_ax.contourf(
            self.theta_bvc.T,
            self.r_bvc.T,
            np.reshape(np.maximum(self.rates.pw, 1e-7), (self.tc_gen.n_bvc_theta, self.tc_gen.n_bvc_r)),
            cmap=self.cmap,
            vmin=0, vmax=1,
            extend="both"
        )

    def on_plot(self):
        """
        Executes the plotting logic for the polar wall representation.
        """
        self.plot()

    def on_clean(self):
        """
        Clears the polar plot in preparation for the next update and sets up the axis labels and ticks for better readability.
        """
        self.pw_ax.clear()
        self.pw_ax.set_yticklabels([])
        self.pw_ax.set_xticklabels([])
        self.pw_ax.grid(color=self.grid_color)
        self.pw_ax.set_theta_zero_location('E')
        self.pw_ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
        self.pw_ax.set_xticklabels(['Right', 'Straight', 'Left', 'Back'])


class BVCPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting Boundary Vector Cell (BVC) representations in an agent-based learning simulation.

    This callback visualizes the agent's BVC activations in a polar coordinate system, providing insights into the agent's spatial cognition and environmental boundaries perception.

    Attributes:
        Requires various parameters from the cache to plot the BVC representation, including a dedicated axis for polar plotting and data related to the agent's spatial cognition and the environment.
    """
    def __init__(self, cmap: str | mcolors.Colormap = 'coolwarm'):
        """
        Initializes the BVCPlotter instance with a specified colormap for the polar plot.

        Args:
            cmap (str | mcolors.Colormap, optional): The colormap for the polar plot. Defaults to 'coolwarm'.
        """
        super().__init__()
        self.cmap = cmap
        self.grid_color = get_most_visible_color(self.cmap)

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the polar plot axis for BVC representation, and specifies the required cache keys for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['bvc_ax'] = cache.fig.add_subplot(cache.gc[:4, 4:8], projection='polar')
        self.requires = [
            'bvc_ax',
            'rates',
            'env',
            'walls_pw',
            'tc_gen',
            'theta_bvc',
            'r_bvc'
        ]
        super().set_cache(cache, on_repeat)

    def plot(self):
        """
        Plots the ego-centric representation of walls and objects.
        """
        self.bvc_ax.contourf(
            self.theta_bvc.T,
            self.r_bvc.T,
            np.reshape(np.maximum(self.rates.bvc, 1e-7), (self.tc_gen.n_bvc_theta, self.tc_gen.n_bvc_r)),
            cmap=self.cmap,
            vmin=0, vmax=1,
            extend="both"
        )

    def on_plot(self):
        """
        Executes the plotting logic for the BVC representation.
        """
        self.plot()

    def on_clean(self):
        """
        Clears the polar plot in preparation for the next update and sets up the axis labels and ticks for navigational directions.
        """
        self.bvc_ax.clear()
        self.bvc_ax.set_yticklabels([])
        self.bvc_ax.set_xticklabels([])
        self.bvc_ax.grid(color=self.grid_color)
        self.bvc_ax.set_theta_zero_location('S')
        self.bvc_ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
        self.bvc_ax.set_xticklabels(['N', 'E', 'S', 'W'])


class oPWPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the Parietal Window (oPW) for objects in an agent-based learning simulation.

    This callback visualizes the agent's perception of objects within its environment in a polar coordinate system, specifically through the lens of the oPW, providing insights into the agent's object-related spatial cognition.

    Attributes:
        Requires various parameters from the cache to plot the oPW representation, including a dedicated axis for polar plotting and data related to the agent's perception of objects and the environment.
    """
    def __init__(self, cmap: str | mcolors.Colormap = 'coolwarm'):
        """
        Initializes the oPWPlotter instance with a specified colormap for the polar plot.

        Args:
            cmap (str | mcolors.Colormap, optional): The colormap for the polar plot. Defaults to 'coolwarm'.
        """
        super().__init__()
        self.cmap = cmap
        self.grid_color = get_most_visible_color(self.cmap)

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the polar plot axis for oPW representation, and specifies the required cache keys for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['opw_ax'] = cache.fig.add_subplot(cache.gc[4:8, 8:], projection='polar')
        self.requires = [
            'opw_ax',
            'rates',
            'env',
            'walls_pw',
            'tc_gen',
            'theta_bvc',
            'r_bvc'
        ]
        super().set_cache(cache, on_repeat)

    def plot(self):
        """
        Plots the ego-centric representation of walls and objects.
        """
        self.opw_ax.contourf(
            self.theta_bvc.T,
            self.r_bvc.T,
            np.reshape(np.maximum(self.rates.opw, 1e-7), (self.tc_gen.n_bvc_theta, self.tc_gen.n_bvc_r)),
            cmap=self.cmap,
            vmin=0, vmax=1,
            extend="both"
        )

    def on_plot(self):
        """
        Executes the plotting logic for the oPW representation.
        """
        self.plot()

    def on_clean(self):
        """
        Clears the polar plot in preparation for the next update and sets up the axis labels and ticks for navigational directions related to object perception.
        """
        self.opw_ax.clear()
        self.opw_ax.set_yticklabels([])
        self.opw_ax.set_xticklabels([])
        self.opw_ax.grid(color=self.grid_color)
        self.opw_ax.set_theta_zero_location('E')
        self.opw_ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
        self.opw_ax.set_xticklabels(['Right', 'Straight', 'Left', 'Back'])


class OVCPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the Object Vector Cells (OVC) representation in an agent-based learning simulation.

    This callback visualizes the agent's OVC activations in a polar coordinate system, providing insights into the agent's object-related spatial cognition and how it perceives objects within the environment.

    Attributes:
        Requires various parameters from the cache to plot the OVC representation, including a dedicated axis for polar plotting and data related to the agent's perception of objects and the environment.
    """
    def __init__(self, cmap: str | mcolors.Colormap = 'coolwarm'):
        """
        Initializes the OVCPlotter instance with a specified colormap for the polar plot.

        Args:
            cmap (str | mcolors.Colormap, optional): The colormap for the polar plot. Defaults to 'coolwarm'.
        """
        super().__init__()
        self.cmap = cmap
        self.grid_color = get_most_visible_color(self.cmap)

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the polar plot axis for OVC representation, and specifies the required cache keys for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['ovc_ax'] = cache['fig'].add_subplot(cache['gc'][:4, 8:], projection='polar')
        self.requires = [
            'ovc_ax',
            'rates',
            'env',
            'walls_pw',
            'tc_gen',
            'theta_bvc',
            'r_bvc'
        ]
        super().set_cache(cache, on_repeat)

    def plot(self):
        """
        Plots the ego-centric representation of walls and objects.
        """
        self.ovc_ax.contourf(
            self.theta_bvc.T,
            self.r_bvc.T,
            np.reshape(np.maximum(self.rates.ovc, 1e-7), (self.tc_gen.n_bvc_theta, self.tc_gen.n_bvc_r)),
            cmap=self.cmap,
            vmin=0, vmax=1,
            extend="both"
        )

    def on_plot(self):
        """
        Executes the plotting logic for the OVC representation.
        """
        self.plot()

    def on_clean(self):
        """
        Clears the polar plot in preparation for the next update and sets up the axis labels and ticks for navigational directions related to object perception.
        """
        self.ovc_ax.clear()
        self.ovc_ax.set_yticklabels([])
        self.ovc_ax.set_xticklabels([])
        self.ovc_ax.grid(color=self.grid_color)
        self.ovc_ax.set_theta_zero_location('S')
        self.ovc_ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
        self.ovc_ax.set_xticklabels(['N', 'E', 'S', 'W'])


class HDPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting Head Direction (HD) cell activations in an agent-based learning simulation.

    This callback visualizes the agent's HD cell activations, providing insights into the agent's directional orientation and spatial cognition.

    Attributes:
        cmap (str): The colormap used for plotting HD cell activations. Defaults to 'coolwarm'.
        theta (np.ndarray): The angular positions for each HD cell activation.
        kwargs (dict): Additional keyword arguments for plotting.
    """
    def __init__(self, cmap: str | mcolors.Colormap = 'coolwarm', **kwargs):
        """
        Initializes the HDPlotter instance with a specified colormap and additional plotting arguments.

        Args:
            cmap (str | mcolors.Colormap, optional): The colormap for HD cell activations. Defaults to 'coolwarm'.
            **kwargs: Arbitrary keyword arguments for plotting.
        """
        super().__init__()
        self.theta = None
        self.cmap = cmap
        self.kwargs = kwargs

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the plotting axis for HD representation, and specifies the required cache keys for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['hd_ax'] = cache['fig'].add_subplot(cache['gc'][8:, 8:])
        self.requires = [
            'hd_ax',
            'rates'
        ]
        super().set_cache(cache, on_repeat)
        self.theta = np.linspace(0, 2*np.pi, len(self.rates.hd), endpoint=False)

    def plot(self):
        """
        Plots the HD cell activations using a scatter plot to visualize the agent's directional orientation.
        """
        for i, (shift, r) in enumerate(zip((1, 8, 4), (1, .95, .9))):

            vis = self.rates.hd.copy()
            if i == 1:
                th_indices = np.where(vis > .1)[0]

                breakdown = np.where(np.diff(th_indices) != 1)[0]
                if breakdown.size:
                    th_indices = np.concatenate([th_indices[breakdown[0]+1:], th_indices[:breakdown[0]+1], ])

                prev = vis[((i)+th_indices[len(th_indices)//2])%len(vis)]
                for i in range(len(th_indices)//2):
                    this_index = ((i)+th_indices[len(th_indices)//2])%len(vis)
                    next_index = (this_index + 1)%len(vis)
                    temp = vis[next_index].copy()
                    vis[next_index] = prev
                    prev = temp.copy()

            self.hd_ax.scatter(
                r*np.cos(self.theta - 2*np.pi/shift),
                r*np.sin(self.theta - 2*np.pi/shift),
                marker='.',
                c=np.roll(vis, int(1/shift*len(self.rates.hd))),
                s=1.5*r*(plt.rcParams['lines.markersize']**2),
                cmap=self.cmap,
                **self.kwargs
            )

    def on_plot(self):
        """
        Executes the plotting logic for the HD cell activations.
        """
        self.plot()

    def on_clean(self):
        """
        Clears the plot in preparation for the next update and sets up the axis for better readability and orientation indication.
        """
        self.hd_ax.clear()
        self.hd_ax.set_axis_off()
        self.hd_ax.set_aspect('equal')
        self.hd_ax.text(1.2, 0, '0°', ha='center', va='center')  # Add a label at the top
        self.hd_ax.text(0, 1.2, '90°', ha='center', va='center')  # Add a label at the top
        self.hd_ax.text(-1.2, 0, '180°', ha='center', va='center')  # Add a label at the top
        self.hd_ax.text(0, -1.2, '270°', ha='center', va='center')  # Add a label at the top
        self.hd_ax.vlines(0, -.8, .8, color='tab:grey', linewidth=1, alpha=.8)
        self.hd_ax.hlines(0, -.8, .8, color='tab:grey', linewidth=1, alpha=.8)
        self.hd_ax.plot(.2*np.cos(self.theta), .2*np.sin(self.theta), color='tab:grey', linewidth=1, alpha=.8)
        self.hd_ax.plot(.5*np.cos(self.theta), .5*np.sin(self.theta), color='tab:grey', linewidth=1, alpha=.8)


class PCPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting Place Cell (PC) activations in an agent-based learning simulation.

    This callback visualizes the agent's PC activations, providing insights into the agent's spatial location and cognitive map of the environment.

    Attributes:
        shape (tuple): The shape of the grid to which PC activations are mapped.
    """
    def __init__(self, cmap: str | mcolors.Colormap = 'coolwarm'):
        """
        Initializes the PCPlotter instance.

        Args:
            cmap (str | mcolors.Colormap, optional): The colormap for PC activations. Defaults to 'coolwarm'.
        """
        self.shape = None
        self.cmap = cmap
        super().__init__()

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the plotting axis for PC representation, and specifies the required cache keys for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        cache['gc_ax'] = cache.fig.add_subplot(cache.gc[8:, :4])
        self.requires = [
            'gc_ax',
            'rates',
            'grid2cart'
        ]
        super().set_cache(cache, on_repeat)
        self.shape = self.cache.grid2cart.shape[0], self.cache.grid2cart.shape[1]

    def plot(self):
        """
        Plots the PC activations using an image plot to visualize the agent's spatial location within the environment.
        """
        self.gc_ax.imshow(
            np.reshape(self.rates.h, self.shape).T,
            origin='lower',
            cmap=self.cmap,
            vmin=0, vmax=1,
        )

    def on_plot(self):
        """
        Executes the plotting logic for the PC activations.
        """
        self.plot()

    def on_clean(self):
        """
        Clears the plot in preparation for the next update and sets up the axis for better readability.
        """
        self.gc_ax.clear()
        self.gc_ax.set_axis_off()


class oPRPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the activations of perirhinal identity neurons (oPR) for objects in an agent-based learning simulation.

    This callback visualizes the oPR neuron activations, providing insights into the agent's recognition and encoding of objects based on their identities.

    Attributes:
        color_new (str): The color used to indicate newly encountered objects. Defaults to 'tab:blue'.
        color_enc (str): The color used to indicate previously encoded objects. Defaults to 'tab:red'.
        labels (list): A list of labels for the objects, derived from their textures.
    """
    def __init__(self, color_new: str = 'tab:blue', color_enc: str = 'tab:red'):
        """
        Initializes the oPRPlotter instance with specified colors for newly encountered and previously encoded objects.

        Args:
            color_new (str, optional): The color for newly encountered objects. Defaults to 'tab:blue'.
            color_enc (str, optional): The color for previously encoded objects. Defaults to 'tab:red'.
        """
        super().__init__()
        self.labels = []
        self.color_enc = color_enc
        self.color_new = color_new

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, initializes the plotting axis for oPR representation, and specifies the required cache keys for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'env',
            'fig',
            'gc',
            'rates',
            'encoding_params',
            'opr_ax'
        ]
        cache['opr_ax'] = cache.fig.add_subplot(cache.gc[8:11, 4:8])
        super().set_cache(cache, on_repeat=on_repeat)

        for obj in self.env.objects:
            self.labels.append(
                obj.polygon.texture.name
            )

    def plot(self):
        """
        Plots the oPR activations using an image plot to visualize objects that were or were not incoded in agent's memory.
        """
        encoded_indices = np.where(self.encoding_params.encoded_objects.ovc.to.h)[0]

        for i, opr_rate in enumerate(self.rates.opr):

            rect = plt.Rectangle(
                (i-.4, 0),
                .8,
                1.1,
                color=self.color_enc if i in encoded_indices else self.color_new,
                alpha=0.3,
                linewidth=0
            )
            self.opr_ax.add_patch(rect)

            rect_border = plt.Rectangle(
                (i - 0.2, 0),
                0.4, opr_rate[0],
                fill=False,
                edgecolor='tab:grey',
                linewidth=1,
                linestyle='--',
                hatch='\\\\\\' if i in encoded_indices else '///'
            )
            self.opr_ax.add_patch(rect_border)

    def on_plot(self):
        """
        Executes the plotting logic for the oPR neuron activations, using different visual cues for newly encountered and previously encoded objects.
        """
        self.plot()


    def on_clean(self):
        """
        Clears the plot in preparation for the next update and sets up the axis for better readability and object identification.
        """
        self.opr_ax.clear()
        self.opr_ax.set_ylim(0, 1)
        self.opr_ax.set_xlim(-.5, len(self.rates.opr)-.5)
        self.opr_ax.set_xticks(range(len(self.labels)), self.labels, rotation=45, ha='right')
        self.opr_ax.set_yticks([])
        for spine_name, spine in self.opr_ax.spines.items():
            if spine_name in ('bottom', 'top'):
                spine.set_edgecolor('tab:grey')
            elif spine_name in ('left', 'right'):
                spine.set_visible(False)


class PickedObjectPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for highlighting picked (selected) objects in an agent-based learning simulation.

    This callback visualizes the selection of objects by the user, providing a visual cue for which objects are currently focused or interacted with.

    Attributes:
        Requires various parameters from the cache to identify and highlight selected objects within the simulation environment.
    """
    def __init__(self, color: str = 'b', **kwargs):
        """
        Initializes the PickedObjectPlotter instance with a specified color and line width for highlighting selected objects.

        Args:
            color (str, optional): The color used to highlight selected objects. Defaults to 'b'.
            **kwargs: Arbitrary keyword arguments to be passed to plot_polygon function.
        """
        super().__init__()
        self.color = color
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('linewidth', 5)
        self.kwargs = kwargs

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting picked objects, and prepares for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['alo_ax', 'click_params', 'env']
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for highlighting picked objects, using visual cues such as border color and thickness.
        """
        if self.click_params['inside_object'] is not False:
            for obj_ind in self.click_params['inside_object']:
                plot_polygon(self.env.objects[obj_ind].polygon, ax=self.alo_ax, color=self.color, zorder=-1, **self.kwargs)


class DistanceAttentionPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for visualizing the attention radius of an agent in an agent-based learning simulation.

    This callback draws a circle around the agent to represent the distance threshold within which objects are considered for attention, aiding in understanding the spatial scope of the agent's attention mechanism.

    Attributes:
        dist_threshold (float): The distance threshold for the agent's attention.
        resolution (int): The resolution of the circle representing the attention radius.
        theta (np.ndarray): The angular coordinates used to calculate the circle's perimeter.
        x_r (np.ndarray): The x-coordinates of the circle's perimeter.
        y_r (np.ndarray): The y-coordinates of the circle's perimeter.
    """
    def __init__(self, dist_threshold: float, resolution: int = 100, color: str = 'r', **kwargs):
        """
        Initializes the DistanceAttentionPlotter instance with a specified distance threshold and resolution for the attention circle.

        Args:
            dist_threshold (float): The distance threshold for the agent's attention.
            resolution (int, optional): The resolution of the circle representing the attention radius. Defaults to 100.
            color (str, optional): The color used to plot the attention radius. Defaults to 'r'.
            **kwargs: Arbitrary keyword arguments to be passed to plot function.
        """
        super().__init__()
        self.dist_threshold = dist_threshold
        self.theta = np.linspace(0, 2*np.pi, resolution)
        self.x_r = dist_threshold * np.cos(self.theta)
        self.y_r = dist_threshold * np.sin(self.theta)
        self.color = color
        kwargs.setdefault('linestyle', ':')
        kwargs.setdefault('linewidth', 1)
        self.kwargs = kwargs

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting the attention radius, and prepares for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = ['alo_ax', 'ego_ax', 'movement_params', 'env']
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the attention radius, drawing circles around the agent in both allocentric and egocentric views.
        """
        ego_circle = np.stack([
            self.x_r + self.movement_params.position[0],
            self.y_r + self.movement_params.position[1]
        ], axis=1)

        alo_circle = np.stack([
            self.x_r,
            self.y_r
        ], axis=1)

        self.alo_ax.plot(ego_circle[:, 0], ego_circle[:, 1], color=self.color, **self.kwargs)
        self.ego_ax.plot(alo_circle[:, 0], alo_circle[:, 1], color=self.color, **self.kwargs)


class MentalAgentPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the mental representation of the agent's position and direction during recall or top-down processing in an agent-based learning simulation.

    This callback visualizes the agent's mental position and direction, providing insights into the agent's internal state and cognitive processes during different modes of operation.

    Attributes:
        Requires various parameters from the cache to plot the mental representation of the agent's position and direction.
    """
    def __init__(self, color: str = '#8fbbd9'):
        """
        Initializes the MentalAgentPlotter instance with a specified color for the mental representation of the agent.

        Args:
            color (str, optional): The color used to plot the mental representation of the agent. Defaults to '#8fbbd9'.
        """
        super().__init__()
        self.color = color

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting the mental representation of the agent's position and direction, and prepares for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'rates',
            'alo_ax',
            'env',
            'weights',
            'dynamics_params',
            'mental_movement_params'
        ]
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the mental representation of the agent's position and direction, using visual cues such as color and markers.
        """
        if self.dynamics_params['mode'] in ('recall', 'top-down'):
            if self.mental_movement_params['position'] is not None:
                self.alo_ax.plot(
                    *self.mental_movement_params['position'],
                    color=self.color,
                    marker='o',
                    zorder=4.5
                )
                if self.mental_movement_params['direction'] is not None:
                    self.alo_ax.arrow(
                        *self.mental_movement_params['position'],
                        0.5 * math.cos(self.mental_movement_params['direction'] ),
                        0.5 * math.sin(self.mental_movement_params['direction'] ),
                        zorder=4.5
                    )


class MentalTargetPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the mental representation of movement and rotation targets during recall or top-down processing in an agent-based learning simulation.

    This callback visualizes the agent's mental targets for movement and rotation, providing insights into the agent's intended actions based on its cognitive processes.

    Attributes:
        move_target_color (str): The color used to plot the mental movement target. Defaults to '#eea8a9'.
        rotate_target_color (str): The color used to plot the mental rotation target. Defaults to '#95cf95'.
    """
    def __init__(
        self,
        move_target_color: str = '#eea8a9',
        rotate_target_color: str = '#95cf95'
    ):
        """
        Initializes the MentalTargetPlotter instance with specified colors for mental movement and rotation targets.

        Args:
            move_target_color (str, optional): The color for mental movement targets. Defaults to '#eea8a9'.
            rotate_target_color (str, optional): The color for mental rotation targets. Defaults to '#95cf95'.
        """
        super().__init__()
        self.move_target_color = move_target_color
        self.rotate_target_color = rotate_target_color

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting the mental targets, and prepares for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'alo_ax',
            'mental_movement_params'
        ]
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the mental representation of movement and rotation targets, using specified colors for each.
        """
        if self.mental_movement_params.move_target is not None:
            self.alo_ax.plot(
                *self.mental_movement_params.move_target,
                'x', color=self.move_target_color,
                zorder=3.5
            )
        if self.mental_movement_params.rotate_target is not None:
            self.alo_ax.plot(
                *self.mental_movement_params.rotate_target,
                'x', color=self.rotate_target_color,
                zorder=3.5
            )


class MentalTargetPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the mental representation of movement and rotation targets during recall or top-down processing in an agent-based learning simulation.

    This callback visualizes the agent's mental targets for movement and rotation, providing insights into the agent's intended actions based on its cognitive processes.

    Attributes:
        move_target_color (str): The color used to plot the mental movement target. Defaults to '#eea8a9'.
        rotate_target_color (str): The color used to plot the mental rotation target. Defaults to '#95cf95'.
    """
    def __init__(
        self,
        move_target_color: str = '#eea8a9',
        rotate_target_color: str = '#95cf95'
    ):
        """
        Initializes the MentalTargetPlotter instance with specified colors for mental movement and rotation targets.

        Args:
            move_target_color (str, optional): The color for mental movement targets. Defaults to '#eea8a9'.
            rotate_target_color (str, optional): The color for mental rotation targets. Defaults to '#95cf95'.
        """
        super().__init__()
        self.move_target_color = move_target_color
        self.rotate_target_color = rotate_target_color

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting the mental targets, and prepares for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'alo_ax',
            'mental_movement_params'
        ]
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the mental representation of movement and rotation targets, using specified colors for each.
        """
        if self.mental_movement_params.move_target is not None:
            self.alo_ax.plot(
                *self.mental_movement_params.move_target,
                'x', color=self.move_target_color,
                zorder=3.5
            )
        if self.mental_movement_params.rotate_target is not None:
            self.alo_ax.plot(
                *self.mental_movement_params.rotate_target,
                'x', color=self.rotate_target_color,
                zorder=3.5
            )


class MentalTrajectoryPlotter(ArtistCallback):
    """
    A specialized ArtistCallback for plotting the mental representation of the agent's trajectory and movement targets during recall or top-down processing in an agent-based learning simulation.

    This callback visualizes the agent's mental trajectory and intended movement targets, providing insights into the agent's cognitive planning and navigational strategy.

    Attributes:
        traj_color (str): The color used to plot the mental trajectory. Defaults to 'tab:green'.
        target_color (str): The color used to plot the final mental movement target. Defaults to 'tab:red'.
    """
    def __init__(self, traj_color: str = 'tab:green', target_color: str = 'tab:red'):
        """
        Initializes the MentalTrajectoryPlotter instance with specified colors for the mental trajectory and movement targets.

        Args:
            traj_color (str, optional): The color for the mental trajectory. Defaults to 'tab:green'.
            target_color (str, optional): The color for the final mental movement target. Defaults to 'tab:red'.
        """
        super().__init__()
        self.traj_color = traj_color
        self.target_color = target_color

    def set_cache(self, cache: Mapping, on_repeat: str = 'raise'):
        """
        Sets the cache with the provided mapping, specifies the required cache keys for plotting the mental trajectory and targets, and prepares for plotting.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (str): The behavior when a cache key is already an attribute. Defaults to 'raise'.
        """
        self.requires = [
            'alo_ax',
            'mental_movement_params',
            'mental_movement_schedule',
            'mental_trajectory'
        ]
        super().set_cache(cache, on_repeat)

    def on_plot(self):
        """
        Executes the plotting logic for the mental representation of the agent's trajectory and movement targets, using specified colors for each.
        """
        if self.mental_movement_params.position is not None and \
            (not len(self.mental_trajectory) or
            not (
                self.mental_movement_params.move_target is not None
                and self.mental_movement_params.move_target not in self.mental_trajectory
            )):
            first_points = [self.mental_movement_params.position, self.mental_movement_params.move_target]\
                if self.mental_movement_params.move_target not in self.mental_movement_schedule\
                and self.mental_movement_params.move_target is not None\
                else [self.mental_movement_params.position]
            all_points = first_points + self.mental_movement_schedule

            if len(self.mental_movement_schedule):
                self.alo_ax.plot(
                    self.mental_movement_schedule[-1][0],
                    self.mental_movement_schedule[-1][1],
                    'X', color=self.target_color,
                    zorder=3.5
                )

            for from_, to in zip(all_points[:-1], all_points[1:]):
                self.alo_ax.plot(*zip(from_, to), '-', color=self.traj_color, alpha=.5, zorder=2.5)

