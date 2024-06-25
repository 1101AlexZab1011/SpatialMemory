from collections import OrderedDict
from dataclasses import dataclass
import logging
import numpy as np
from typing import Callable, Literal, Mapping
import shapely as spl
import shapely.prepared as splp
from shapely import Polygon, Point
from bbtoolkit.utils.math.geometry import compute_intersection3d, create_cartesian_space, create_shapely_points, find_closest_points, poly2vectors, regroup_min_max
from bbtoolkit.utils.math.tensor_algebra import sub3d
from bbtoolkit.environment import Environment, Object, SpatialParameters
from bbtoolkit.environment.builders import EnvironmentBuilder
from bbtoolkit.environment.compilers.callbacks import BaseCompilerCallback
from bbtoolkit.environment.compilers.structures import EnvironmentMetaData
from bbtoolkit.environment.visible_planes import LazyVisiblePlane, PrecomputedVisiblePlane
from bbtoolkit.structures import BaseCallbacksManager
from bbtoolkit.structures.geometry import TexturedPolygon
from bbtoolkit.utils.indextools import remove_slice


class EnvironmentCompiler:
    """
    A class that compiles an environment using an EnvironmentBuilder.

    Attributes:
        builder (EnvironmentBuilder): The builder used to compile the environment.
    """
    def __init__(
        self,
        builder: EnvironmentBuilder,
        visible_plane_compiler: Callable[
            [
                np.ndarray | list[np.ndarray],
                np.ndarray | list[np.ndarray],
                np.ndarray,
                list[np.ndarray]
            ],
            np.ndarray,
        ] = None,
    ):
        """
        Initializes the EnvironmentCompiler with an EnvironmentBuilder.

        Args:
            builder (EnvironmentBuilder): The builder used to compile the environment.
        """
        self._builder = builder
        self._visible_plane_compiler = visible_plane_compiler

    @property
    def builder(self) -> EnvironmentBuilder:
        """
        Returns the builder used to compile the environment.

        Returns:
            EnvironmentBuilder: The builder used to compile the environment.
        """
        return self._builder

    @property
    def visible_plane_compiler(self) -> Callable[
        [
            np.ndarray | list[np.ndarray],
            np.ndarray | list[np.ndarray],
            np.ndarray,
            list[np.ndarray]
        ],
        np.ndarray,
    ]:
        """
        Returns the visible plane compiler.

        Returns:
            Callable: The visible plane compiler.
        """
        return self._visible_plane_compiler

    def compile_room_area(self) -> Polygon:
        """
        Compiles the room area boundaries into a Polygon.

        Returns:
            Polygon: The room area.
        """
        return Polygon([
            Point(self.builder.xy_min, self.builder.xy_min),
            Point(self.builder.xy_min, self.builder.xy_max),
            Point(self.builder.xy_max, self.builder.xy_max),
            Point(self.builder.xy_max, self.builder.xy_min)
        ])

    def compile_visible_area(self) -> Polygon:
        """
        Compiles the visible area boundaries into a Polygon.

        Returns:
            Polygon: The visible area.
        """
        return Polygon([
            Point(self.builder.x_train_min, self.builder.y_train_min),
            Point(self.builder.x_train_min, self.builder.y_train_max),
            Point(self.builder.x_train_max, self.builder.y_train_max),
            Point(self.builder.x_train_max, self.builder.y_train_min)
        ])

    def compile_space_points(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ):
        """
        Compiles the space points from a range of coordinates.

        Args:
            from_ (tuple[int, int]): The starting coordinates.
            to (tuple[int, int]): The ending coordinates.

        Returns:
            list[Point]: The compiled space points.
        """
        return create_shapely_points(
            x_coords, y_coords,
            res=self.builder.res
        )

    @staticmethod
    def compile_room_points(
        space_points: list[Point],
        objects: list[Polygon]
    ) -> list[Point]:
        """
        Compiles the room points from a list of space points.

        Args:
            space_points (list[Point]): The space points.
            objects (list[Polygon]): The objects.

        Returns:
            list[Point]: The compiled room points.
        """
        prepared = splp.prep(spl.GeometryCollection([poly.obj for poly in objects]))
        return list(filter(prepared.disjoint, space_points))

    @staticmethod
    def compile_visible_area_points(
        room_points: list[Point],
        visible_area: Polygon
    ) -> list[Point]:
        """
        Compiles the visible area points from a list of room points and a visible area.

        Args:
            room_points (list[Point]): The room points.
            visible_area (Polygon): The visible area.

        Returns:
            list[Point]: The compiled visible area points.
        """
        prepared = splp.prep(visible_area)
        return list(filter(prepared.contains, room_points))

    @staticmethod
    def compile_boundary_points(
        space_points: list[Point],
        objects: list[Polygon]
    ) -> list[Point]:
        """
        Compiles the boundary points from a list of space points.

        Args:
            space_points (list[Point]): The space points.

        Returns:
            list[Point]: The compiled boundary points.
        """
        prepared_objects_boundaries = [
            splp.prep(obj.boundary)
            for obj in objects
        ]
        return [
            val
            for prepared in prepared_objects_boundaries
            if len(val := list(filter(prepared.crosses, space_points)))
        ]

    @staticmethod
    def align_objects(
        grid_points: list[np.ndarray],
        objects: list[Polygon]
    ) -> list[TexturedPolygon]:
        """
        Aligns the object polygons with the resolution of the space grid.

        Args:
            grid_points (np.ndarray): The space grid points.

        Returns:
            list[TexturedPolygon]: The aligned object boundaries.

        Notes:
            Alignment is done accordingly to the given boundary points. Shapes of resulting objects can be incorrect.
        """
        # points of space are centers of circles of r=0.5*res.
        # These centers sometimes are not consistent with boundaries of original objects.
        # So we need to make correction to be consistent with resolution
        object_matrices_exterior = [
            np.array(obj.exterior.coords.xy).T
            for obj in objects
        ]

        object_matrices_exterior_corrected = [
            find_closest_points(points, object_matrix)
            for object_matrix, points in zip(object_matrices_exterior, grid_points)
        ]
        object_matrices_interior = [
            [np.array(interior.coords.xy).T for interior in obj.interiors]
            for obj in objects
        ]
        object_matrices_interior_corrected = [
            [
                find_closest_points(points, object_matrix)
                for object_matrix in interior
            ] if interior else None
            for interior, points in zip(object_matrices_interior, grid_points)
        ]
        return [ # redefine objects according to space grid correction
            TexturedPolygon(
                zip(
                    obj_coords_exterior[:, 0],
                    obj_coords_exterior[:, 1]
                ),
                [
                    zip(hole[:, 0], hole[:, 1])
                    for hole in obj_holes
                ] if obj_holes else None,
                texture=obj.texture
            )
            for obj,
            obj_coords_exterior,
            obj_holes
            in zip(
                objects,
                object_matrices_exterior_corrected,
                object_matrices_interior_corrected
            )
        ]

    @staticmethod
    def compile_visible_plane(
        starting_points: np.ndarray | list[np.ndarray],
        directions: np.ndarray | list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray] # list of boundary points coordinates for each object
    ) -> LazyVisiblePlane | PrecomputedVisiblePlane:
        """
        Compiles the visible plane from starting points, directions, room points coordinates, and boundary points coordinates.

        Args:
            starting_points (np.ndarray): Coordinates of each vertex for all objects. Can be list of vertex coordinates for each object.
            directions (np.ndarray): The vertex-wise differences representing direction vectors from one vertex to another.
            room_points_coordinates (np.ndarray): The room points coordinates.
            boundary_points_coordinates (list[np.ndarray]): The boundary points coordinates for each object.

        Returns:
            VisiblePlane | PrecomputedVisiblePlane: The compiled visible plane.
        """
        starting_points = np.concatenate(starting_points) if isinstance(starting_points, list) else starting_points
        directions = np.concatenate(directions) if isinstance(directions, list) else directions
        all_boundary_points_coordinates = np.concatenate(boundary_points_coordinates)

        n_boundary_points = len(all_boundary_points_coordinates)
        n_training_points = len(room_points_coordinates)

        directions = np.concatenate( # add z coordinate with zeros to directions
            [
                directions,
                np.zeros((*directions.shape[:-1], 1))
            ],
            axis=-1
        )
        local_starting_points = sub3d( # each starting point minus each point of room area
            starting_points,
            room_points_coordinates,
            return_2d=False
        )
        local_starting_points = np.concatenate( # add z coordinate with zeros to local starting points
            [
                local_starting_points,
                np.zeros((*local_starting_points.shape[:-1], 1))
            ],
            axis=-1
        )

        local_boundary_points = sub3d( # each boundary point minus each point of room area
            all_boundary_points_coordinates,
            room_points_coordinates,
            return_2d=False
        )
        local_boundary_points = np.concatenate( # add z coordinate with zeros to local boundary points
            [
                local_boundary_points,
                np.zeros((*local_boundary_points.shape[:-1], 1))
            ],
            axis=-1
        )

        alpha_pt, alpha_occ = compute_intersection3d( # compute intersection points between each line and each boundary using cross product
            np.zeros_like(local_boundary_points), # starting point of each line is [0, 0, 0] (egocentric location of agent)
            local_starting_points, # starting points of each line is each point of object relative to egocentric location of agent
            local_boundary_points, # direction of each line is each boundary point relative to egocentric location of agent
            np.repeat(directions[np.newaxis, :, :], n_training_points, axis=0) # direction of each line is distanec from one vertex of an object to another
        )

        mask = ~np.any((alpha_pt < 1 - 1e-5) & (alpha_pt > 0) & (alpha_occ < 1) & (alpha_occ > 0), axis=1)

        visible_xy = np.full((n_training_points, n_boundary_points, 2), np.nan)
        for location, location_mask in enumerate(mask):
            visible_xy[location, location_mask] = all_boundary_points_coordinates[location_mask]

        cumulative_lengths = np.cumsum([len(boundary) for boundary in boundary_points_coordinates])
        slices = [slice(from_, to) for from_, to in zip([0] + list(cumulative_lengths[:-1]), cumulative_lengths)]

        return PrecomputedVisiblePlane(visible_xy, slices)

    @staticmethod
    def compile_directions(objects: list[Polygon]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compiles the directions from a list of objects.

        Args:
            objects (list[Polygon]): The objects.

        Returns:
            tuple[np.ndarray, np.ndarray]: The compiled directions.
        """
        starting_points, directions = list(), list()
        for obj in objects:
            starting_points_, directions_ = poly2vectors(obj)
            starting_points.append(starting_points_)
            directions.append(directions_)

        return starting_points, directions

    @staticmethod
    def compile_objects(
        space_points: list[Point],
        space_points_coordinates: np.ndarray,
        visible_coordinates: np.ndarray,
        objects: list[Polygon],
        visible_plane_compiler: Callable[
            [
                np.ndarray | list[np.ndarray],
                np.ndarray | list[np.ndarray],
                np.ndarray,
                list[np.ndarray]
            ],
                np.ndarray,
            ] = None
    ) -> list[Object]:
        """
        Compiles objects with visible parts from each space point

        Args:
            space_points (list[Point]): The list of space points
            space_points_coordinates (np.ndarray): Coordinates of all points in space
            visible_coordinates (np.ndarray): Coordinates of points outside objects
            objects (list[Polygon]): Polygon objects representing shapes of objects
            visible_plane_compiler (Callable): A function to compute visible parts of each object from each location.
                If None, EnvironmentCompiler.compile_visible_plane is used. See EnvironmentCompiler.compile_visible_plane for details. Default is None.

        Returns:
            list[Object]: The list of compiled objects.
        """
        objects_corrected = EnvironmentCompiler.align_objects(
            [space_points_coordinates for _ in range(len(objects))],
            objects
        )
        boundary_points = EnvironmentCompiler.compile_boundary_points(space_points, objects_corrected)
        boundary_points_coordinates = [
            np.array([[point.centroid.xy[0][0], point.centroid.xy[1][0]] for point in boundary_point])
            for boundary_point in boundary_points
        ]

        starting_points, directions = EnvironmentCompiler.compile_directions(objects_corrected)
        visible_plane_compiler = EnvironmentCompiler.compile_visible_plane \
            if visible_plane_compiler is None else visible_plane_compiler
        visible_plane = visible_plane_compiler(
            starting_points,
            directions,
            visible_coordinates,
            boundary_points_coordinates
        )
        return [
            Object(
                obj,
                boundary_points_coordinates[i],
                visible_plane[i],
                starting_points[i],
                directions[i]
            )
            for i, obj in enumerate(objects_corrected)
        ]

    def compile(
        self
    ) -> Environment:
        """
        Compiles the environment.

        Returns:
            Environment: The compiled environment.
        """
        room_area = self.compile_room_area()
        visible_area = self.compile_visible_area()

        x_coords, y_coords = create_cartesian_space(
            *regroup_min_max(*visible_area.bounds),
            self.builder.res
        )

        space_points = self.compile_space_points(
            x_coords, y_coords
        )
        space_points_coordinates = np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in space_points
        ])

        visible_space_points = self.compile_room_points(
            space_points,
            self.builder.walls + self.builder.objects
        )
        visible_space_points_coordinates = np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in visible_space_points
        ])

        visible_objects = self.compile_objects(
            space_points,
            space_points_coordinates,
            visible_space_points_coordinates,
            self.builder.objects + self.builder.walls,
            self._visible_plane_compiler
        )
        return Environment(
            room_area,
            visible_area,
            visible_objects[:len(self.builder.objects)],
            visible_objects[len(self.builder.objects):],
            SpatialParameters(
                self.builder.res,
                (x_coords, y_coords),
                visible_space_points_coordinates
            )
        )


class DynamicEnvironmentCompiler(EnvironmentCompiler, BaseCallbacksManager):
    """
    A class used to compile dynamic environments.

    Attributes:
        space_points (list[Point]): The space points.
        space_points_coordinates (np.ndarray): The coordinates of the space points.
        visible_plane (VisiblePlane): The visible plane.
        environment (Environment): The compiled environment.
        objects_metadata (list[EnvironmentMetaData]): The metadata of the objects in the environment.
        callbacks (list[BaseCompilerCallback]): A list of callbacks to be called on events. Defaults to None.

    Methods:
        get_n_vertices(polygon: Polygon) -> int: Returns the number of vertices in a given polygon.
        compile_visible_coordinates(objects: list[Polygon]): Compiles the coordinates of the visible space points.
        _get_entity_container(entity_type: str) -> list[TexturedPolygon]: Returns the list of entities of a given type.
        _add_entity(entities: list[TexturedPolygon], entities_type: Literal['object', 'wall']): Adds a list of entities of a given type to the environment.
        add_object(*objects: TexturedPolygon): Adds a list of objects to the environment.
        remove_entity(index: int, entity_type: Literal['object', 'wall']): Removes an entity of a given type from the environment.
        remove_object(index: int): Removes an object from the environment.
        add_wall(*walls: TexturedPolygon): Adds a list of walls to the environment.
        remove_wall(index: int): Removes a wall from the environment.
    """
    def __init__(
        self,
        builder: EnvironmentBuilder,
        visible_plane_compiler: Callable[
            [
                np.ndarray | list[np.ndarray],
                np.ndarray | list[np.ndarray],
                np.ndarray,
                list[np.ndarray]
            ],
            np.ndarray,
        ] = None,
        callbacks: list[BaseCompilerCallback] = None,
        cache: Mapping = None
    ):
        """
        The constructor for DynamicEnvironmentCompiler class.

        Args:
            builder (EnvironmentBuilder): An instance of the EnvironmentBuilder class used to build the environment.
            visible_plane_compiler (Callable, optional): A function used to compile the visible plane. If None, LazyVisiblePlane is used. Defaults to None.
            callbacks (list[BaseCompilerCallback], optional): A callback or a list of callbacks to be called on events. Defaults to None.
        """
        visible_plane_compiler = visible_plane_compiler if visible_plane_compiler is not None else LazyVisiblePlane
        EnvironmentCompiler.__init__(self, builder, visible_plane_compiler)
        room_area = self.compile_room_area()
        visible_area = self.compile_visible_area()
        x_coords, y_coords = create_cartesian_space(
            *regroup_min_max(*visible_area.bounds),
            self.builder.res
        )
        self.space_points = self.compile_space_points(
            x_coords, y_coords
        )
        self.space_points_coordinates = np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in self.space_points
        ])

        visible_space_points_coordinates = self.compile_visible_coordinates(
            self.builder.walls + self.builder.objects
        )
        visible_objects = self.compile_objects(
            self.space_points,
            self.space_points_coordinates,
            visible_space_points_coordinates,
            self.builder.objects + self.builder.walls,
            self._visible_plane_compiler
        )
        self.visible_plane = visible_objects[0].visible_parts.visible_plane
        self.environment = Environment(
            room_area,
            visible_area,
            visible_objects[:len(self.builder.objects)],
            visible_objects[len(self.builder.objects):],
            SpatialParameters(
                self.builder.res,
                (x_coords, y_coords),
                visible_space_points_coordinates
            )
        )
        vertices = [
            self.get_n_vertices(obj.polygon) - 1 - 1*len(obj.polygon.interiors) # -1 because the last point is the same as the first one
            for obj in self.environment.objects + self.environment.walls
        ]
        cumulative_lengths = np.cumsum(vertices)
        vectors_slices = [slice(from_, to) for from_, to in zip([0] + list(cumulative_lengths[:-1]), cumulative_lengths)]

        self.objects_metadata = [
            EnvironmentMetaData('object', self.visible_plane.slices[i], vectors_slices[i])
            for i, _ in enumerate(self.builder.objects)
        ]
        self.objects_metadata += [ # objects go first in EnvironmentCompiler
            EnvironmentMetaData('wall', self.visible_plane.slices[i_], vectors_slices[i_])
            for i_ in range(
                len(self.builder.objects),
                len(self.builder.objects) + len(self.builder.walls)
            )
        ]

        if cache is None:
            cache = dict(compiler=self)
        else:
            cache.update(dict(compiler=self))

        BaseCallbacksManager.__init__(self, callbacks, cache)

    @staticmethod
    def get_n_vertices(polygon: Polygon) -> int:
        """
        Static method to get the number of vertices in a given polygon.

        Args:
            polygon (Polygon): The polygon whose vertices are to be counted.

        Returns:
            int: The number of vertices in the polygon.
        """
        return len(polygon.exterior.coords) +\
            sum([
                len(interior.coords) for interior in polygon.interiors
            ] if len(polygon.interiors) > 0 else [0])

    def compile_visible_coordinates(
        self, objects: list[Polygon]
    ):
        """
        Method to compile the coordinates of the visible boundaries points.

        Args:
            objects (list[Polygon]): A list of polygons representing the objects in the environment.

        Returns:
            np.array: An array of coordinates of the visible space points.
        """
        visible_space_points = self.compile_room_points(
            self.space_points,
            objects
        )
        return np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in visible_space_points
        ])

    def _get_entity_container(self, entity_type: str) -> list[TexturedPolygon]:
        """
        Method to get the list of entities of a given type.

        Args:
            entity_type (str): The type of the entities to be returned. Can be 'object' or 'wall'.

        Returns:
            list[TexturedPolygon]: A list of entities of the given type.
        """
        if entity_type == 'object':
            return self.environment.objects
        elif entity_type == 'wall':
            return self.environment.walls

    def _add_entity(
        self,
        entities: list[TexturedPolygon],
        entities_type: Literal['object', 'wall']
    ):
        """
        Method to add a list of entities of a given type to the environment.

        Args:
            entities (list[TexturedPolygon]): A list of entities to be added to the environment.
            entities_type (Literal['object', 'wall']): The type of the entities to be added. Can be 'object' or 'wall'.
        """
        self.visible_plane.room_points_coordinates = self.compile_visible_coordinates(
            [obj.polygon for obj in self.environment.walls + self.environment.objects] + list(entities)
        )
        entities_corrected = self.align_objects(
            [self.space_points_coordinates for _ in range(len(entities))],
            entities
        )

        boundary_points = self.compile_boundary_points(self.space_points, entities_corrected)
        boundary_points_coordinates = [
            np.array([[point.centroid.xy[0][0], point.centroid.xy[1][0], 0] for point in boundary_point])
            for boundary_point in boundary_points
        ]
        # adding new boundary points to the existing ones
        self.visible_plane.boundary_points = np.concatenate([self.visible_plane.boundary_points, *boundary_points_coordinates])


        cumulative_lengths = np.cumsum([len(boundary) for boundary in boundary_points_coordinates])
        # adding new slices for the new objects
        self.visible_plane.slices += [
            slice(from_ + self.visible_plane.slices[-1].stop, to + self.visible_plane.slices[-1].stop)
            for from_, to in zip([0] + list(cumulative_lengths[:-1]), cumulative_lengths)
        ]

        # adding new starting points and directions
        starting_points, directions = self.compile_directions(entities_corrected)
        starting_points_concat, directions_concat = np.concatenate(starting_points), np.concatenate(directions)
        self.visible_plane.starting_points = np.concatenate([
            self.visible_plane.starting_points,
            np.concatenate( # add z coordinate with zeros to local starting points
                [
                    starting_points_concat,
                    np.zeros((*starting_points_concat.shape[:-1], 1))
                ],
                axis=-1
            )
        ])
        self.visible_plane.directions = np.concatenate([
            self.visible_plane.directions,
            np.concatenate( # add z coordinate with zeros to local starting points
                [
                    directions_concat,
                    np.zeros((*directions_concat.shape[:-1], 1))
                ],
                axis=-1
            )
        ])

        # cleaning cache
        self.visible_plane.cache_manager.cache = OrderedDict()
        n_existing_entities = len(self.environment.objects) + len(self.environment.walls)

        entities_container = self._get_entity_container(entities_type)

        for i, obj in enumerate(entities_corrected):
            entities_container.append(
                Object(
                    obj,
                    boundary_points_coordinates[i],
                    self.visible_plane[i + n_existing_entities],
                    starting_points[i],
                    directions[i]
                )
            )
            self.objects_metadata.append(
                EnvironmentMetaData(
                    entities_type,
                    self.visible_plane.slices[i + n_existing_entities],
                    slice(self.objects_metadata[-1].vec_slice.stop, self.objects_metadata[-1].vec_slice.stop + len(starting_points[i]))
                )
            )

        self.environment.params.coords = self.visible_plane.room_points_coordinates

        updated = slice(n_existing_entities, n_existing_entities + len(entities))

        self.callbacks.execute('on_change', updated, self.objects_metadata[updated])
        self.callbacks.execute('on_add', updated)

        if entities_type not in ('object', 'wall'):
            logging.warning(f'Unexpected entities type: {entities_type}')

        self.callbacks.execute(f'on_add_{entities_type}', updated)

    def add_object(self, *objects: TexturedPolygon):
        """
        Method to add a list of objects to the environment.

        Args:
            objects (TexturedPolygon): The objects to be added to the environment.
        """
        self._add_entity(objects, 'object')

    def _remove_entity(self, index: int, entity_type: Literal['object', 'wall']):
        """
        Method to remove an entity of a given type from the environment.

        Args:
            index (int): The index of the entity to be removed.
            entity_type (Literal['object', 'wall']): The type of the entity to be removed. Can be 'object' or 'wall'.
        """
        entities_container = self._get_entity_container(entity_type)

        if index < 0:
            obj_index = len(entities_container) + index
        else:
            obj_index = index

        n_entities = -1
        for i, obj in enumerate(self.objects_metadata):
            if obj.type == entity_type:
                n_entities += 1

                if n_entities == obj_index:
                    abs_index = i

        self.visible_plane.room_points_coordinates = self.compile_visible_coordinates(
            [obj.polygon for obj in self.environment.walls + self.environment.objects if obj != entities_container[obj_index]]
        )
        # removing boundary points of the object
        self.visible_plane.boundary_points = np.concatenate([
            self.visible_plane.boundary_points[:self.objects_metadata[abs_index].vp_slice.start],
            self.visible_plane.boundary_points[self.objects_metadata[abs_index].vp_slice.stop:]
        ])
        # removing slices of the object
        self.visible_plane.slices = remove_slice(self.visible_plane.slices, abs_index)
        # removing starting points and directions of the object
        self.visible_plane.starting_points = np.concatenate([
            self.visible_plane.starting_points[:self.objects_metadata[abs_index].vec_slice.start],
            self.visible_plane.starting_points[self.objects_metadata[abs_index].vec_slice.stop:]
        ])
        self.visible_plane.directions = np.concatenate([
            self.visible_plane.directions[:self.objects_metadata[abs_index].vec_slice.start],
            self.visible_plane.directions[self.objects_metadata[abs_index].vec_slice.stop:]
        ])
        # cleaning cache
        self.visible_plane.cache_manager.cache = OrderedDict()
        # Remove object from the list
        if entity_type == 'object':
            self.environment.objects = self.environment.objects[:obj_index] + self.environment.objects[obj_index + 1:]
        elif entity_type == 'wall':
            self.environment.walls = self.environment.walls[:obj_index] + self.environment.walls[obj_index + 1:]

        for obj in self.environment.walls + self.environment.objects:
            if obj.visible_parts.object_index > abs_index:
                obj.visible_parts.object_index -= 1

        # Remove object from the metadata
        vp_slices = remove_slice([meta.vp_slice for meta in self.objects_metadata], abs_index)
        vec_slices = remove_slice([meta.vec_slice for meta in self.objects_metadata], abs_index)
        removed_meta = self.objects_metadata[abs_index].copy()
        self.objects_metadata = self.objects_metadata[:abs_index] + self.objects_metadata[abs_index + 1:]

        for meta, vp_slice, vec_slice in zip(self.objects_metadata, vp_slices, vec_slices):
            meta.vp_slice = vp_slice
            meta.vec_slice = vec_slice

        self.environment.params.coords = self.visible_plane.room_points_coordinates

        self.callbacks.execute('on_remove', abs_index, removed_meta)
        self.callbacks.execute('on_change', abs_index, removed_meta)

        if entity_type not in ('object', 'wall'):
            logging.warning(f'Unexpected entities type: {entity_type}')

        self.callbacks.execute(f'on_remove_{entity_type}', abs_index, removed_meta)

    def remove_object(self, index: int):
        """
        Method to remove an object from the environment.

        Args:
            index (int): The index of the object to be removed.
        """
        self._remove_entity(index, 'object')

    def add_wall(self, *walls: TexturedPolygon):
        """
        Method to add a list of walls to the environment.

        Args:
            walls (TexturedPolygon): The walls to be added to the environment.
        """
        self._add_entity(walls, 'wall')

    def remove_wall(self, index: int):
        """
        Method to remove a wall from the environment.

        Args:
            index (int): The index of the wall to be removed.
        """
        self._remove_entity(index, 'wall')
