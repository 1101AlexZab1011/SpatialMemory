from abc import ABC, abstractmethod
from collections import OrderedDict
import numbers
from typing import Awaitable, Generator

import numpy as np
from sklearn.neighbors import KDTree

from bbtoolkit.data import Cached, Copyable, asynchronous
from bbtoolkit.math.geometry import compute_intersection3d, get_closest_points_indices
from bbtoolkit.utils import remove_slice


class AbstractVisiblePlaneSubset(Copyable, ABC):
    """
    Represents an abstract class for subset of a visible plane.
    This class is used to define the interface for accesing visible points for particular object.
    """
    @abstractmethod
    def __getitem__(self, indices: int | tuple[int, int] | tuple[int, int, int]) -> np.ndarray: # position, points, axis
        """
        Allows the VisiblePlaneSubset object to be indexed. First index is the position, second is the points, third is the axis.
        """
        pass
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of room points coordinates in the VisiblePlaneSubset.
        """
        pass
    @abstractmethod
    def __iter__(self) -> np.ndarray:
        """
        Allows iteration over points of the VisiblePlaneSubset object.
        """
        pass
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the VisiblePlaneSubset (number of positions in space, points of object boundaries, x or y axis).
        """
        pass


class AbstractVisiblePlane(Copyable, ABC):
    """
    Base class for visible plane in a 2D space.
    """
    @abstractmethod
    def __getitem__(self, i: int) -> np.ndarray | AbstractVisiblePlaneSubset:
        """
        Allows the VisiblePlane object to be indexed. Each index corresponds to a different object.
        """
        pass


class PrecomputedVisiblePlane(AbstractVisiblePlane):
    """
    A class representing a visible plane in a 3D space.

    Attributes:
        _data (np.ndarray): The visible coordinates of the plane.
        _slices (list[slice]): The slices for each object in the visible coordinates.
    """
    def __init__(
        self,
        visible_coordinates: np.ndarray, # shape (n_locations, n_boundary_points, 2)
        object_slices: list[slice], # list of slices for each object in visible_coordinates
    ):
        """
        Initializes the VisiblePlane with visible coordinates and object slices.

        Args:
            visible_coordinates (np.ndarray): The visible coordinates of the plane.
                Shape is (n_locations, n_boundary_points, 2).
            object_slices (list[slice]): The slices for each object in the visible coordinates.
        """
        self._data = visible_coordinates
        self._slices = object_slices

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Returns the visible coordinates for the object at the given index.

        Args:
            i (int): The index of the object.

        Returns:
            np.ndarray: The visible coordinates for the object.
        """
        return self._data[:, self._slices[i], :]

    @property
    def data(self) -> np.ndarray:
        """
        Returns the visible coordinates of the plane.

        Returns:
            np.ndarray: The visible coordinates of the plane.
        """
        return self._data

    @property
    def slices(self) -> list[slice]:
        """
        Returns the slices for each object in the visible coordinates.

        Returns:
            list[slice]: The slices for each object.
        """
        return self._slices


class VisiblePlaneSubset(AbstractVisiblePlaneSubset):
    """
    A class that represents a subset of a visible plane in a 3D space.

    Attributes:
        visible_plane (VisiblePlane): The visible plane object that this subset belongs to.
        object_index (int): The index of the object in the visible plane.
    """
    def __init__(
        self,
        visible_plane: 'LazyVisiblePlane',
        object_index: int
    ):
        """
        Constructs all the necessary attributes for the VisiblePlaneSubset object.

        Args:
            visible_plane (VisiblePlane): The visible plane object that this subset belongs to.
            object_index (int): The index of the object in the visible plane.
        """
        self.visible_plane = visible_plane
        self.object_index = object_index

    def __call__(
        self,
        coords_x: float,
        coords_y: float
    ) -> np.ndarray:
        """
        Makes the VisiblePlaneSubset object callable.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            np.ndarray: The visible coordinates for the object.
        """
        # still efficient since LazyVisiblePlane.__call__ is cached
        return self.visible_plane(coords_x, coords_y)[self.object_index]

    def __getitem__(self, indices: int | tuple[int, int] | tuple[int, int, int]) -> np.ndarray: # position, points, axis
        """
        Allows the VisiblePlaneSubset object to be indexed.

        Args:
            indices (int | tuple[int, int] | tuple[int, int, int]): The indices to access.

        Returns:
            np.ndarray: The accessed elements.
        """
        if isinstance(indices, tuple):
            position_index = indices[0]
            rest_indices = indices[1:]
        else:
            position_index = indices
            rest_indices = ()

        coords_x = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 0])
        coords_y = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 1])

        return np.concatenate([self.visible_plane(x, y)[self.object_index] for x, y in zip(coords_x, coords_y)])[*rest_indices] # points, axis

    def __len__(self) -> int:
        """
        Returns the number of room points coordinates in the VisiblePlaneSubset.

        Returns:
            int: Number of room points coordinates.
        """
        return len(self.visible_plane.room_points_coordinates)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Allows iteration over the VisiblePlaneSubset object.

        Yields:
            Generator[np.ndarray, None, None]: The accessed elements.
        """
        for index in range(len(self)):
            yield self[index]

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the VisiblePlaneSubset.

        Returns:
            tuple[int, ...]: A tuple representing the shape of the VisiblePlaneSubset.
        """
        return self.visible_plane.room_points_coordinates.shape[0],\
            self.visible_plane.slices[self.object_index].stop - self.visible_plane.slices[self.object_index].start,\
            self.visible_plane.room_points_coordinates.shape[-1]


class LazyVisiblePlane(AbstractVisiblePlane):
    """
    A class that represents a visible plane in a 2D space.

    Attributes:
        room_points_coordinates (np.ndarray): Coordinates of the room points.
        slices (list[slice]): List of slices for boundary points.
        boundary_points (np.ndarray): Coordinates of the boundary points.
        starting_points (np.ndarray): Starting points for the plane.
        directions (np.ndarray): Directions for the plane.
        cache_manager (Cached): Cache manager for the plane.
    """
    def __init__(
        self,
        starting_points: np.ndarray | list[np.ndarray],
        directions: np.ndarray | list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray],
        cache_manager: Cached = None
    ):
        """
        Constructs all the necessary attributes for the LazyVisiblePlane object.

        Args:
            starting_points (np.ndarray | list[np.ndarray]): Starting points for the plane.
            directions (np.ndarray | list[np.ndarray]): Directions for the plane.
            room_points_coordinates (np.ndarray): Coordinates of the room points.
            boundary_points_coordinates (list[np.ndarray]): Coordinates of the boundary points.
            cache_manager (Cached, optional): Cache manager for the plane. Defaults to None.
        """
        self.room_points_coordinates = room_points_coordinates

        cumulative_lengths = np.cumsum([len(boundary) for boundary in boundary_points_coordinates])
        self.slices = [slice(from_, to) for from_, to in zip([0] + list(cumulative_lengths[:-1]), cumulative_lengths)]
        boundary_points = np.concatenate(boundary_points_coordinates)
        self.boundary_points = np.concatenate( # add z coordinate with zeros to local boundary points
            [
                boundary_points,
                np.zeros((*boundary_points.shape[:-1], 1))
            ],
            axis=-1
        )

        starting_points = np.concatenate(starting_points) if isinstance(starting_points, list) else starting_points
        self.starting_points = np.concatenate( # add z coordinate with zeros to local starting points
            [
                starting_points,
                np.zeros((*starting_points.shape[:-1], 1))
            ],
            axis=-1
        )

        directions = np.concatenate(directions) if isinstance(directions, list) else directions
        self.directions = np.concatenate( # add z coordinate with zeros to directions
            [
                directions,
                np.zeros((*directions.shape[:-1], 1))
            ],
            axis=-1
        )
        self.cache_manager = cache_manager if cache_manager is not None else Cached()


    @staticmethod
    def _process_visible_coordinates(
        coords_x: float,
        coords_y: float,
        starting_points: np.ndarray,
        directions: np.ndarray,
        boundary_points: np.ndarray,
        slices: list[slice]
    ) -> list[np.ndarray]:
        """
        Compute all visible points from the given coordinate.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.
            starting_points (np.ndarray): Starting points for each object in the plane.
            directions (np.ndarray): Pointwise difference between pairs of verteces of each object.
            boundary_points (np.ndarray): Coordinates of the boundary points.
            slices (list[slice]): List of slices for boundary points describing coordinates of each object.

        Returns:
            list[np.ndarray]: List of visible xy coordinates.
        """
        local_starting_points = starting_points - np.array([[coords_x, coords_y, 0]])
        local_boundary_points = boundary_points - np.array([[coords_x, coords_y, 0]])
        alpha_pt, alpha_occ = compute_intersection3d(
            np.zeros_like(local_boundary_points),
            local_starting_points,
            local_boundary_points,
            directions
        )
        mask = ~np.any((alpha_pt < 1 - 1e-5) & (alpha_pt > 0) & (alpha_occ < 1) & (alpha_occ > 0), axis=0)
        visible_xy = np.full((len(local_boundary_points), 2), np.nan)

        visible_xy[mask] = boundary_points[mask, :2]

        return [
            visible_xy[slice_]
            for slice_ in slices
        ]


    def __call__(
        self,
        coords_x: float,
        coords_y: float
    ) -> list[np.ndarray]:
        """
        Makes the VisiblePlane object callable.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            list[np.ndarray]: List of visible xy coordinates for each object.
        """
        @self.cache_manager
        def nested_call(
            coords_x: float,
            coords_y: float
        ) -> list[np.ndarray]:
            return self._process_visible_coordinates(
                coords_x, coords_y,
                self.starting_points, self.directions,
                self.boundary_points, self.slices
            )

        return nested_call(coords_x, coords_y)

    def __getitem__(self, index: int) -> VisiblePlaneSubset:
        """
        Allows the LazyVisiblePlane object to be indexed.

        Args:
            index (int): Index of the desired slice.

        Returns:
            VisiblePlaneSubset: A subset of the VisiblePlane.
        """
        return VisiblePlaneSubset(self, index)

    def __len__(self) -> int:
        """
        Returns the number of slices in the LazyVisiblePlane.

        Returns:
            int: Number of slices.
        """
        return len(self.slices)

    def __iter__(self) -> Generator[VisiblePlaneSubset, None, None]:
        """
        Allows iteration over the VisiblePlane object.

        Yields:
            Generator[VisiblePlaneSubset, None, None]: A subset of the LazyVisiblePlane.
        """
        for index in range(len(self)):
            yield VisiblePlaneSubset(self, index)


class AsyncVisiblePlaneSubset(VisiblePlaneSubset):
    """
    A class that represents an asynchronous subset of a visible plane in a 2D space.

    This class inherits from the VisiblePlaneSubset class and overrides some of its methods
    to provide asynchronous functionality.

    Attributes:
        visible_plane (VisiblePlane): The visible plane object that this subset belongs to.
        object_index (int): The index of the object in the visible plane.
    """
    def __getitem__(self, indices: int | tuple[int, int] | tuple[int, int, int]): # position, points, axis
        if isinstance(indices, tuple):
            position_index = indices[0]
            rest_indices = indices[1:]
        else:
            position_index = indices
            rest_indices = ()

        if isinstance(position_index, numbers.Integral):
            closest_points_indices = set(get_closest_points_indices(
                self.visible_plane.room_points_coordinates,
                position_index,
                tree=self.visible_plane.tree,
                n_points=self.visible_plane.n_neighbours + 1 # + 1 to include the point itself
            )) - {position_index}
            res = self.visible_plane( # validate presence in cache
                self.visible_plane.room_points_coordinates[position_index, 0],
                self.visible_plane.room_points_coordinates[position_index, 1]
            )

            for i in closest_points_indices:
                _ = self.visible_plane( # validate presence in cache
                    self.visible_plane.room_points_coordinates[i, 0],
                    self.visible_plane.room_points_coordinates[i, 1]
                )
            return res.result()[self.object_index][*rest_indices]

        coords_x = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 0])
        coords_y = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 1])

        return np.concatenate([self.visible_plane(x, y).result()[self.object_index] for x, y in zip(coords_x, coords_y)])[*rest_indices] # points, axis

    def __len__(self) -> int:
        """
        Returns the number of room points coordinates in the AsyncVisiblePlaneSubset.

        Returns:
            int: Number of room points coordinates.
        """
        return len(self.visible_plane.room_points_coordinates)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Allows iteration over the AsyncVisiblePlaneSubset object.

        Yields:
            Generator[np.ndarray, None, None]: The accessed elements.
        """
        for index in range(len(self)):
            yield self[index]

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the AsyncVisiblePlaneSubset.

        Returns:
            tuple[int, ...]: A tuple representing the shape of the AsyncVisiblePlaneSubset.
        """
        return self.visible_plane.room_points_coordinates.shape[0],\
            self.visible_plane.slices[self.object_index].stop - self.visible_plane.slices[self.object_index].start,\
            self.visible_plane.room_points_coordinates.shape[-1]


class AsyncVisiblePlane(LazyVisiblePlane):
    """
    A class that represents a visible plane in a 2D space.

    This class inherits from the LazyisiblePlane class and overrides some of its methods
    to provide lazy asynchronous functionality.

    Attributes:
        n_neighbours (int): The number of neighbours to consider.
        tree (KDTree): The KDTree for efficient nearest neighbour search.
    """
    def __init__(
        self,
        starting_points: np.ndarray | list[np.ndarray],
        directions: np.ndarray | list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray], # list of boundary points coordinates for each object
        cache_manager: Cached = None,
        n_neighbours: int = 10,
    ):
        """
        Constructs all the necessary attributes for the AsyncVisiblePlane object.

        Args:
            starting_points (np.ndarray | list[np.ndarray]): Starting points for the plane.
            directions (np.ndarray | list[np.ndarray]): Directions for the plane.
            room_points_coordinates (np.ndarray): Coordinates of the room points.
            boundary_points_coordinates (list[np.ndarray]): Coordinates of the boundary points.
            cache_manager (Cached, optional): Cache manager for the plane. Defaults to None.
            n_neighbours (int, optional): The number of neighbours to consider. Defaults to 10.
        """
        if cache_manager is None:
            cache_manager = Cached(cache_storage=OrderedDict())

        super().__init__(
            starting_points,
            directions,
            room_points_coordinates,
            boundary_points_coordinates,
            cache_manager
        )
        self.n_neighbours = n_neighbours
        self.tree = KDTree(room_points_coordinates)

    def __call__(
        self,
        coords_x: float,
        coords_y: float
    ) -> Awaitable[list[np.ndarray]]:
        """
        Makes the AsyncVisiblePlane object callable.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            Awaitable[list[np.ndarray]]: List of visible xy coordinates for each object.
        """
        @self.cache_manager
        @asynchronous
        def nested_call(
            coords_x: float,
            coords_y: float
        ) -> list[np.ndarray]:
            return self._process_visible_coordinates(
                coords_x, coords_y,
                self.starting_points, self.directions,
                self.boundary_points, self.slices
            )

        return nested_call(coords_x, coords_y)

    def __getitem__(self, index: int) -> AsyncVisiblePlaneSubset:
        """
        Allows the LasyAsyncVisiblePlane object to be indexed.

        Args:
            index (int): Index of the desired slice.

        Returns:
            AsyncVisiblePlaneSubset: An asynchronous subset of the LasyAsyncVisiblePlane.
        """
        return AsyncVisiblePlaneSubset(self, index)


class LazyVisiblePlaneWithTransparancy(LazyVisiblePlane):
    """
    A class that extends LazyVisiblePlane to handle transparent objects in visible planes.

    Attributes:
        transparent_indices (list[int]): Indices of objects that are transparent.
        opaque_indices (list[int]): Indices of objects that are opaque.
        transparent_vectors_slices (list[slice]): Slices corresponding to vertexes of transparent objects (in starting_points and directions).
        opaque_vectors_slices (list[slice]): Slices corresponding to  vertexes of opaque objects (in starting_points and directions).

    Args:
        starting_points (list[np.ndarray]): List of starting points for the vectors.
        directions (list[np.ndarray]): List of direction vectors (vertex-wise differences).
        room_points_coordinates (np.ndarray): Array of room points coordinates.
        boundary_points_coordinates (list[np.ndarray]): List of objects boundary points coordinates.
        transparent (list[int], optional): List of indices that are transparent. Defaults to None.
        cache_manager (Cached, optional): Cache manager instance to use for caching. Defaults to None.
    """
    def __init__(
        self,
        starting_points: list[np.ndarray],
        directions: list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray],
        transparent: list[int] = None,
        cache_manager: Cached = None
    ):
        """
        Initializes the LazyVisiblePlaneWithTransparancy instance.
        """
        super().__init__(
            starting_points,
            directions,
            room_points_coordinates,
            boundary_points_coordinates,
            cache_manager
        )
        self.transparent_indices = transparent if transparent is not None else []
        self.opaque_indices = np.setdiff1d(np.arange(len(starting_points)), transparent).tolist()
        slices = self.create_slices(directions)
        self.transparent_vectors_slices = slices[self.transparent_indices].tolist() if len(self.transparent_indices) != 0 else []
        self.opaque_vectors_slices = slices[self.opaque_indices].tolist() if len(self.opaque_indices) != 0 else []

    def track_new(self, vector_slice: slice, transparent: bool = False):
        """
        Tracks a new vector slice as either transparent or opaque.

        Args:
            vector_slice (slice): The slice of the vector to track.
            transparent (bool, optional): Flag indicating if the slice is transparent. Defaults to False.
        """
        container = self.transparent_indices if transparent else self.opaque_indices
        container.append(max(self.opaque_indices + self.transparent_indices) + 1)
        vectors_slices = self.transparent_vectors_slices if transparent else self.opaque_vectors_slices
        vectors_slices.append(vector_slice)

    def set_opaque(self, index: int):
        """
        Sets a previously transparent vector slice to opaque.

        Args:
            index (int): The index of the vector slice to set as opaque.
        """
        if index in self.opaque_indices:
            return
        num = self.transparent_indices.index(index)
        self.opaque_indices.append(self.transparent_indices.pop(num))
        self.opaque_vectors_slices.append(self.transparent_vectors_slices.pop(num))

    def set_transparent(self, index: int):
        """
        Sets a previously opaque vector slice to transparent.

        Args:
            index (int): The index of the vector slice to set as transparent.
        """
        if index in self.transparent_indices:
            return
        num = self.opaque_indices.index(index)
        self.transparent_indices.append(self.opaque_indices.pop(num))
        self.transparent_vectors_slices.append(self.opaque_vectors_slices.pop(num))

    def untrack(self, index: int):
        """
        Untracks a vector slice when corresponding object is removed.

        Args:
            index (int): The index of the vector slice to untrack.

        Raises:
            ValueError: If the index is not currently tracked.
        """
        all_indices = sorted(self.transparent_indices + self.opaque_indices)
        all_slices = sorted(self.transparent_vectors_slices + self.opaque_vectors_slices)

        if index not in all_indices:
            raise ValueError(f'Index {index} is not tracked')

        all_slices = remove_slice(all_slices, index)
        all_slices.insert(index, None)
        all_indices[index] = None

        #FIXME: DRY principle is violated here
        new_transparent_indices, new_transparent_slices = list(), list()
        for i in self.transparent_indices:
            if all_indices[i] is not None:
                new_transparent_indices.append(all_indices[i]) if \
                    all_indices[i] < index else new_transparent_indices.append(all_indices[i] - 1)
                new_transparent_slices.append(all_slices[i])

        self.transparent_indices = new_transparent_indices
        self.transparent_vectors_slices = new_transparent_slices

        new_opaque_indices, new_opaque_slices = list(), list()
        for i in self.opaque_indices:
            if all_indices[i] is not None:
                new_opaque_indices.append(all_indices[i]) if \
                    all_indices[i] < index else new_opaque_indices.append(all_indices[i] - 1)
                new_opaque_slices.append(all_slices[i])

        self.opaque_indices = new_opaque_indices
        self.opaque_vectors_slices = new_opaque_slices

    @staticmethod
    def create_slices(arraylist: list[np.ndarray]) -> list[slice]:
        """
        Creates a list of slices based on the lengths of arrays in the input list of arrays.

        Args:
            arraylist (list[np.ndarray]): List of arrays to create slices from.

        Returns:
            list[slice]: List of slices for the input arrays.
        """
        lengths = [len(arr) for arr in arraylist]
        slices = [slice(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
        return np.array(slices)

    @staticmethod
    def slicemask(array: np.ndarray, slices: list[slice]) -> np.ndarray:
        """
        Creates a mask for an array based on the given slices.

        Args:
            array (np.ndarray): The array to mask.
            slices (list[slice]): The slices to use for masking.

        Returns:
            np.ndarray: The masked array.
        """
        mask = np.zeros(array.shape[0], dtype=bool)

        for sl in slices:
            mask[sl] = True

        return array[mask]

    def __call__(
        self,
        coords_x: float,
        coords_y: float
    ) -> list[np.ndarray]:
        """
        Makes the LazyVisiblePlaneWithTransparancy object callable and calculates visible points from the given coordinates.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            list[np.ndarray]: List of visible xy coordinates for each object.
        """
        @self.cache_manager
        def nested_call(
            coords_x: float,
            coords_y: float,
            starting_points, directions,
            boundary_points, slices
        ) -> list[np.ndarray]:
            return self._process_visible_coordinates(
                coords_x, coords_y,
                starting_points, directions,
                boundary_points, slices
            )

        #FIXME: Not efficient and vulnerable code
        hard_plane = nested_call(
            coords_x, coords_y,
            self.slicemask(self.starting_points, self.opaque_vectors_slices),
            self.slicemask(self.directions, self.opaque_vectors_slices),
            self.boundary_points,
            self.slices
        )
        hard_visible_boundary_points = np.concatenate([
            np.concatenate(hard_plane),
            np.zeros((len(self.boundary_points), 1))
        ], -1)
        for transparent_slice, transparent_index in zip(
            self.transparent_vectors_slices,
            self.transparent_indices
        ):
            hard_plane[transparent_index] = nested_call(
                coords_x, coords_y,
                self.starting_points[transparent_slice],
                self.directions[transparent_slice],
                hard_visible_boundary_points,
                [self.slices[transparent_index]]
            )[0]

        return hard_plane
