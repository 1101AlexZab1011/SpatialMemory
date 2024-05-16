import configparser
from matplotlib import pyplot as plt
from shapely import Polygon, Point
from shapely.validation import explain_validity
from bbtoolkit.utils.data import Copyable
from bbtoolkit.utils.data.configparser import EvalConfigParser
from bbtoolkit.utils.viz import plot_polygon
from bbtoolkit.structures.geometry import Texture, TexturedPolygon


class EnvironmentBuilder(Copyable):
    """
    A class for building environments, defining training areas, objects, and creating configurations.

    Attributes:
        xy_min (float): Minimum value for X and Y axes of the environment.
        xy_max (float): Maximum value for X and Y axes of the environment.
        xy_train_min (float | tuple[float, float]): Minimum training area coordinates for X and Y (default is None).
        xy_train_max (float | tuple[float, float]): Maximum training area coordinates for X and Y (default is None).
        res (float): The resolution used for processing geometry data (default is 0.3).

    Methods:
        to_config(self) -> configparser.ConfigParser: Convert the environment configuration to a ConfigParser object.
        save(self, path: str): Save the environment configuration to a file at the specified path.
        load(cls, path: str) -> 'EnvironmentBuilder': Load an environment configuration from a file.
        add_object(self, *args: Object2D) -> 'EnvironmentBuilder': Add objects to the environment.
        plot(self, show: bool = False) -> plt.Figure: Plot the environment.

    Example:
        >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, res=0.5)
        >>> builder.set_textures(5).set_polygons(8)
        >>> builder.add_object(Object2D(x=(0, 1, 1, 0), y=(0, 0, 1, 1)))
        >>> fig = builder.plot(show=True)
    """
    def __init__(
        self,
        xy_min: float,
        xy_max: float,
        xy_train_min: float | tuple[float, float] = None,
        xy_train_max: float | tuple[float, float] = None,
        res: float =  0.3,
    ) -> None:
        # Initialize the EnvironmentBuilder with specified configurations
        self.xy_min = xy_min
        self.xy_max = xy_max

        if xy_train_max is None:
            self.x_train_max, self.y_train_max = self.xy_max, self.xy_max
        elif isinstance(xy_train_max, float):
            self.x_train_max, self.y_train_max = xy_train_max, xy_train_max
        else:
            self.x_train_max, self.y_train_max = xy_train_max
        if xy_train_min is None:
            self.x_train_min, self.y_train_min = self.xy_min, self.xy_min
        elif isinstance(xy_train_min, float):
            self.x_train_min, self.y_train_min = xy_train_min, xy_train_min
        else:
            self.x_train_min, self.y_train_min = xy_train_min

        self.res = res
        self.objects = list()
        self.walls = list()

    @staticmethod
    def _obj2config(config: EvalConfigParser, name: str, obj: TexturedPolygon) -> None:
        """
        Add an object to the configuration.

        Args:
            config (EvalConfigParser): The configuration to which the object will be added.
            name (str): The name of the object.
            obj (TexturedPolygon): The object to be added to the configuration.
        """
        config.add_section(name)
        config.set(name, 'n_vertices', str(len(obj.exterior.xy[0])))
        config.set(name, 'exterior_x', str(obj.exterior.xy[0].tolist())[1:-1])
        config.set(name, 'exterior_y', str(obj.exterior.xy[1].tolist())[1:-1])
        config.set(name, 'interiors_x', str([interior.xy[0].tolist() for interior in obj.interiors]) if obj.interiors else '')
        config.set(name, 'interiors_y', str([interior.xy[1].tolist() for interior in obj.interiors]) if obj.interiors else '')
        config.set(name, 'texture_id', str(obj.texture.id_))
        config.set(name, 'texture_color', f'"{str(obj.texture.color)}"')
        config.set(name, 'texture_name', f'"{str(obj.texture.name)}"')

    def to_config(self) -> configparser.ConfigParser:
        """
        Generate a configuration parser instance containing environmental information.

        Returns:
            configparser.ConfigParser: Configuration parser instance representing the environmental boundaries,
            training area, building boundaries, and object vertices.

        The generated configuration contains sections representing different aspects of the environment:
        - 'ExternalSources': Empty sections for paths and variables.
        - 'GridBoundaries': Contains maximum and minimum XY coordinate and resolution details.
        - 'TrainingRectangle': Describes the training area coordinates.
        - 'BuildingBoundaries': Holds the maximum number of object points, number of objects, and
          counts of polygons and textures in the environment.

        The object-specific information is stored under individual sections 'Object{i}' for each object.
        Each object section contains 'n_vertices' and 'object_x'/'object_y' detailing the object's vertices.
        """
        parser = EvalConfigParser()
        parser.add_section('ExternalSources')
        parser.set('ExternalSources', 'paths', '')
        parser.set('ExternalSources', 'variables', '')

        parser.add_section('GridBoundaries')
        parser.set('GridBoundaries', 'max_xy', str(self.xy_max))
        parser.set('GridBoundaries', 'min_xy', str(self.xy_min))
        parser.set('GridBoundaries', 'res', str(self.res))

        parser.add_section('TrainingRectangle')
        parser.set('TrainingRectangle', 'min_train_x', str(self.x_train_min))
        parser.set('TrainingRectangle', 'min_train_y', str(self.y_train_min))
        parser.set('TrainingRectangle', 'max_train_x', str(self.x_train_max))
        parser.set('TrainingRectangle', 'max_train_y', str(self.y_train_max))

        parser.add_section('BuildingBoundaries')
        # parser.set('BuildingBoundaries', 'max_n_obj_points', str(max([len(obj.exterior.xy[0]) for obj in self.objects])))
        parser.set('BuildingBoundaries', 'n_objects', str(len(self.objects)))

        for i, obj in enumerate(self.objects):
            self._obj2config(parser, f'Object{i+1}', obj)

        parser.set('BuildingBoundaries', 'n_walls', str(len(self.walls)))

        for i, obj in enumerate(self.walls):
            self._obj2config(parser, f'Wall{i+1}', obj)

        return parser

    def save(self, path: str):
        """
        Save the generated environment configuration to a specified .ini file.

        Args:
            path (str): The file path to which the configuration will be saved.

        This method uses the `to_config` method to generate the environment configuration and then writes
        it to a file specified by the 'path' argument.
        """
        config = self.to_config()

        with open(path, 'w') as f:
            config.write(f)

    @staticmethod
    def load(path: str) -> 'EnvironmentBuilder':
        """
        Load an environment configuration from a specified .ini file and create an `EnvironmentBuilder` instance.

        Args:
            path (str): The file path from which the environment configuration will be loaded.

        This method loads the configuration stored in the file specified by the 'path' argument. The loaded
        configuration includes details of the grid boundaries, training rectangle, objects, and building boundaries.
        It then uses this loaded information to create an `EnvironmentBuilder` instance.

        Returns:
            EnvironmentBuilder: An `EnvironmentBuilder` instance with the loaded environment configuration.

        Example:
            >>> builder = EnvironmentBuilder.load('environment_config.ini')
            >>> # The builder variable now contains an `EnvironmentBuilder` instance with the loaded configuration.
        """
        config = EvalConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
        config.read(path)
        return EnvironmentBuilder(
            config['GridBoundaries'].eval('min_xy'),
            config['GridBoundaries'].eval('max_xy'),
            (
                config['TrainingRectangle'].eval('min_train_x'),
                config['TrainingRectangle'].eval('min_train_y')
            ),
            (
                config['TrainingRectangle'].eval('max_train_x'),
                config['TrainingRectangle'].eval('max_train_y')
            ),
            config['GridBoundaries'].eval('res')
        ).add_wall(
            *[
                TexturedPolygon(
                    shell=[
                        Point(x, y)
                        for x, y in zip(
                            config[f'Wall{i}'].eval('exterior_x'),
                            config[f'Wall{i}'].eval('exterior_y')
                        )
                    ],
                    holes=[
                        [
                            Point(x, y)
                            for x, y in zip(
                                interiors_x,
                                interiors_y
                            )
                        ]
                        for interiors_x, interiors_y in zip(
                            config[f'Wall{i}'].eval('interiors_x'),
                            config[f'Wall{i}'].eval('interiors_y')
                        )
                    ] if config[f'Wall{i}'].eval('interiors_x') else None,
                    texture=Texture(
                        config[f'Wall{i}'].eval('texture_id'),
                        config[f'Wall{i}'].eval('texture_color'),
                        config[f'Wall{i}'].eval('texture_name')
                    )
                )
                for i in range(1, config['BuildingBoundaries'].eval('n_walls')+1)
            ]
        ).add_object(
            *[
                TexturedPolygon(
                    shell = (
                        Point(x, y)
                        for x, y in zip(
                            config[f'Object{i}'].eval('exterior_x'),
                            config[f'Object{i}'].eval('exterior_y')
                        )
                    ),
                    texture=Texture(
                        config[f'Object{i}'].eval('texture_id'),
                        config[f'Object{i}'].eval('texture_color'),
                        config[f'Object{i}'].eval('texture_name')
                    )
                )
                for i in range(1, config['BuildingBoundaries'].eval('n_objects')+1)
            ]
        )

    def __validate_objects(self, *objects: Polygon | TexturedPolygon) -> None:
        for object_ in objects:
            if not object_.is_valid:
                raise ValueError(f'Object {object_} is not valid: {explain_validity(object_)}')

    def __validate_textures(self, *objects: Polygon | TexturedPolygon) -> list[TexturedPolygon]:
        out = list()
        for obj in objects:
            if isinstance(obj, TexturedPolygon):
                out.append(obj)
            else:
                out.append(TexturedPolygon(obj))

        return out

    def add_object(self, *args: Polygon | TexturedPolygon) -> 'EnvironmentBuilder':
        """
        Add one or multiple objects to the environment being constructed.

        Args:
            *args (Polygon | TexturedPolygon): Variable number of objects to be added to the environment.

        This method appends one or more objects to the list of objects within the environment being built.
        Each object contain details such as texture and coordinates of the geometric objects present
        within the environment.

        Returns:
            EnvironmentBuilder: The updated instance of the EnvironmentBuilder with the added objects.
        """
        self.__validate_objects(*args)
        self.objects += list(self.__validate_textures(*args))
        return self

    def add_wall(self, *args: Polygon | TexturedPolygon) -> 'EnvironmentBuilder':
        """
        Add one or multiple objects to the environment being constructed.

        Args:
            *args (Polygon | TexturedPolygon): Variable number of objects to be added to the environment.

        This method appends one or more objects to the list of objects within the environment being built.
        Each object contain details such as texture and coordinates of the geometric objects present
        within the environment.

        Returns:
            EnvironmentBuilder: The updated instance of the EnvironmentBuilder with the added objects.
        """
        self.__validate_objects(*args)
        self.walls += list(self.__validate_textures(*args))
        return self

    def remove_object(self, i: int) -> 'EnvironmentBuilder':
        """
        Removes the object at the specified index from the list of objects in the environment.

        Args:
            i (int): The index of the object to be removed.

        Returns:
            EnvironmentBuilder: The modified EnvironmentBuilder object after removing the specified object.
        """
        self.objects.pop(i)
        return self

    def remove_wall(self, i: int) -> 'EnvironmentBuilder':
        """
        Removes the wall at the specified index from the list of objects in the environment.

        Args:
            i (int): The index of the object to be removed.

        Returns:
            EnvironmentBuilder: The modified EnvironmentBuilder object after removing the specified object.
        """
        self.walls.pop(i)
        return self

    def __add__(self, other: 'EnvironmentBuilder') -> 'EnvironmentBuilder':
        """
        Adds the objects and properties of two EnvironmentBuilder instances.

        Merges the objects from two separate EnvironmentBuilder instances into a new instance.
        The new instance retains the original attributes of the first instance (self), such as grid boundaries,
        training rectangle, resolution, and objects. It also appends the objects and updates the properties (textures
        and polygons) as specified.

        Args:
            other (EnvironmentBuilder): Another EnvironmentBuilder instance to be combined with the current one.

        Returns:
            EnvironmentBuilder: A new EnvironmentBuilder instance containing the combined objects and attributes from self and other.
        """
        return EnvironmentBuilder(
            self.xy_min,
            self.xy_max,
            self.x_train_min,
            self.y_train_min,
            self.x_train_max,
            self.y_train_max,
            self.res,
        ).add_object(
            *self.objects,
            *other.objects
        ).add_wall(
            *self.walls,
            *other.walls
        )

    def plot(self, ax: plt.Axes = None) -> plt.Figure:
        """
        Visualizes the environment layout by generating a plot using matplotlib.

        Args:
            ax (plt.Axes, optional): Matplotlib Axes to use for plotting. If None, a new subplot is created. Defaults to None.

        This method generates a plot that visualizes the layout of the environment using matplotlib. It plots the
        boundaries of the entire environment, the training area, and the objects within it.

        Returns:
            plt.Figure: A matplotlib Figure object representing the generated plot.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> fig = builder.plot(show=True)
            >>> # The plot showing the environment layout will be displayed.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # plot whole area
        ax.plot(
            (self.xy_min, self.xy_min, self.xy_max, self.xy_max, self.xy_min),
            (self.xy_min, self.xy_max, self.xy_max, self.xy_min, self.xy_min),
            '-', color='#999', label='Whole Area'
        )
        # plot training area
        ax.plot(
            (self.x_train_min, self.x_train_min, self.x_train_max, self.x_train_max, self.x_train_min),
            (self.y_train_min, self.y_train_max, self.y_train_max, self.y_train_min, self.y_train_min),
            '--', color='tab:blue', label='Training Area'
        )

        # plot walls
        for wall in self.walls:
            if wall.texture.color is None:
                plot_polygon(wall, color='tab:orange', ax=ax)
            else:
                plot_polygon(wall, ax=ax)

        # plot objects
        for obj in self.objects:
            plot_polygon(obj, ax=ax)

        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
        ax.grid()

        return fig