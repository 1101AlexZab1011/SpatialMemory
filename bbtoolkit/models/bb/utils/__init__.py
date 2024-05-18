import numpy as np
from bbtoolkit.environment import Environment
from bbtoolkit.structures.tensorgroups import DirectedTensorGroup, NamedTensor, TensorGroup


def activity2rate(activity: TensorGroup, connectivity: DirectedTensorGroup) -> TensorGroup:
    """
    Converts the activity of a TensorGroup to firing rates using a sigmoid projection.

    Args:
        activity (TensorGroup): The activity of the neurons.
        connectivity (DirectedTensorGroup): The synaptic connectivity between the neurons.

    Returns:
        TensorGroup: The firing rates of the neurons.
    """
    rates = TensorGroup()
    for key, tensor in activity.data.items():
        key_ = 'pr' if 'pr' in key else key # object identity cells, use same beta and alpha as other PR neurons by definition
        rates.add_tensor(
            NamedTensor(
                key,
                1/(
                    1 + np.exp(-2*connectivity[key_, key_]['beta']*(tensor - connectivity[key_, key_]['alpha']))
                )
            )
        )
    return rates


class Grid2CartTransition:
    """
    A class to manage the transition between grid coordinates and Cartesian coordinates within a given environment.

    This class facilitates the conversion of coordinates from a grid-based representation to a Cartesian (x, y) representation and vice versa, based on the environment's parameters.

    Attributes:
        env (Environment): The environment object containing parameters for coordinate conversion.
    """
    def __init__(self, env: Environment):
        """
        Initializes the Grid2CartTransition instance with environment parameters.

        Args:
            env (Environment): The environment object containing parameters for coordinate conversion.
        """
        coords_x, coords_y = env.params.coords[:, 0], env.params.coords[:, 1]
        self.min_train_x, self.max_train_x, self.min_train_y, self.max_train_y = min(coords_x), max(coords_x), min(coords_y), max(coords_y)
        self.shape = int((self.max_train_x - self.min_train_x)/env.params.res), int((self.max_train_y - self.min_train_y)/env.params.res)


    def __call__(self, x: float, y: float) -> tuple[int, int]:
        """
        Converts Cartesian coordinates to grid indices.

        Args:
            x (float): The x-coordinate in Cartesian space.
            y (float): The y-coordinate in Cartesian space.

        Returns:
            tuple[int, int]: The corresponding grid indices.
        """
        x = (x - self.min_train_x)/(self.max_train_x - self.min_train_x)
        y = (y - self.min_train_y)/(self.max_train_y - self.min_train_y)

        return self.shape[1] - int(y*self.shape[1]), int(x*self.shape[0])

    def __getitem__(self, indices: tuple[int, int]) -> tuple[float, float]:
        """
        Converts grid indices to Cartesian coordinates.

        Args:
            indices (tuple[int, int]): The grid indices.

        Returns:
            tuple[float, float]: The corresponding Cartesian coordinates.
        """
        i, j = indices
        x = ((self.shape[0] - i) / self.shape[0]) * (self.max_train_x - self.min_train_x) + self.min_train_x
        # FIXME: y-axis is inverted
        y = ((j) / self.shape[1]) * (self.max_train_y - self.min_train_y) + self.min_train_y
        return y, x # because in a matrix first index goes for columns and second for rows


def get_pr_cue(env: Environment, walls_fov: list[np.ndarray]) -> np.ndarray:
    """
    Generates a perirhinal (PR) cue vector based on the visibility and distance of walls within the field of view.

    Args:
        env (Environment): The simulation environment containing walls and their textures.
        walls_fov (list[np.ndarray]): A list of numpy arrays, each representing the points of a wall that are within the field of view.

    Returns:
        np.ndarray: The PR cue vector, where each element represents a normalized count of points for each texture, adjusted by the minimum distance to the observer.
    """
    # 1. get amount of points for each texture id
    counts = dict()
    distances = dict()
    for wall, wall_fov in zip(env.walls, walls_fov):
        if wall.polygon.texture.id_ not in counts:
            counts[wall.polygon.texture.id_] = len(wall_fov)
            d = np.sqrt(np.sum(wall_fov**2, axis=1))
            if d.size:
                distances[wall.polygon.texture.id_] = np.min(d)
            else:
                distances[wall.polygon.texture.id_] = np.inf
        else:
            counts[wall.polygon.texture.id_] += len(wall_fov)
            distances[wall.polygon.texture.id_] = min(distances[wall.polygon.texture.id_], np.min(np.sqrt(np.sum(wall_fov**2, axis=1))))

    distances = dict(sorted(distances.items(), key=lambda x: x[0])) # walls with smallest id first
    counts = dict(sorted(counts.items(), key=lambda x: x[0]))
    # 2. get the pr cue vector which is counts normalized by the distance
    pr_cue = np.array([count/dist for count, dist in zip(counts.values(), distances.values())])

    return pr_cue
