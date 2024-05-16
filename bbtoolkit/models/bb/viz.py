import logging
from typing import Mapping

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent
import numpy as np
from bbtoolkit.dynamics.callbacks.viz import ArtistCallback
from bbtoolkit.math.geometry import calculate_polar_distance
from bbtoolkit.utils.viz import plot_arrow, plot_polygon
from bbtoolkit.utils.viz.colors import adjust_color_brightness
from shapely import Point


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
        super().__init()
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
            cmap='coolwarm',
            vmin=0, vmax=1
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
            cmap='coolwarm',
            vmin=0, vmax=1
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
            cmap='coolwarm',
            vmin=0, vmax=1
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
            cmap='coolwarm',
            vmin=0, vmax=1
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
    def __init__(self, cmap: str = 'coolwarm', **kwargs):
        """
        Initializes the HDPlotter instance with a specified colormap and additional plotting arguments.

        Args:
            cmap (str, optional): The colormap for HD cell activations. Defaults to 'coolwarm'.
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
        cache['hd_ax'] = cache.fig.add_subplot(cache.gc[8:, 8:])
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
    def __init__(self):
        """
        Initializes the PCPlotter instance.
        """
        self.shape = None
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
        self.shape = self.cache.grid2cart.shape

    def plot(self):
        """
        Plots the PC activations using an image plot to visualize the agent's spatial location within the environment.
        """
        self.gc_ax.imshow(
            np.reshape(self.rates.h, self.shape),
            cmap='coolwarm',
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
