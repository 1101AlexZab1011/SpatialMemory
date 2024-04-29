from dataclasses import dataclass
from typing import Literal, Mapping
from bbtoolkit.data import Copyable
from bbtoolkit.preprocessing.environment.compilers.structures import EnvironmentMetaData
from bbtoolkit.preprocessing.environment.visible_planes import LazyVisiblePlaneWithTransparancy
from bbtoolkit.structures import BaseCallback


class BaseCompilerCallback(BaseCallback):
    """
    A base class for creating callback hooks to respond to various events triggered by the
    DynamicEnvironmentCompiler during the compilation process.

    This class is intended to be subclassed to implement custom behavior for each event.

    Attributes:
        compiler (DynamicEnvironmentCompiler): A reference to the associated compiler instance.
                                               This should be set using the `set_compiler` method.
    """

    def set_cache(self, cache: Mapping):
        """
        Sets the reference to the associated compiler instance.

        Args:
            compiler (DynamicEnvironmentCompiler): The compiler instance to associate with this callback.
        """
        self.requires = ['compiler']
        super().set_cache(cache)

    def on_change(self, i: int | slice, metadata: EnvironmentMetaData | list[EnvironmentMetaData]):
        """
        Called when an existing environment component is changed. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice of the components that have changed.
            metadata (EnvironmentMetaData | list[EnvironmentMetaData]): The metadata associated with
                the changed components.
        """
        ...

    def on_add(self, i: int | slice):
        """
        Called when a new environment component is added. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice where the new components are added.
        """
        ...

    def on_remove(self, i: int | slice, metadata: EnvironmentMetaData | list[EnvironmentMetaData]):
        """
        Called when an existing environment component is removed. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice of the components that are being removed.
            metadata (EnvironmentMetaData | list[EnvironmentMetaData]): The metadata associated with
                the removed components.
        """
        ...

    def on_add_object(self, i: int | slice):
        """
        Called when a new object is added to the environment. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice where the new objects are added.
        """
        ...

    def on_remove_object(self, i: int | slice, metadata: EnvironmentMetaData | list[EnvironmentMetaData]):
        """
        Called when an object is removed from the environment. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice of the objects that are being removed.
            metadata (EnvironmentMetaData | list[EnvironmentMetaData]): The metadata associated with
                the removed objects.
        """
        ...

    def on_add_wall(self, i: int | slice):
        """
        Called when a new wall is added to the environment. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice where the new walls are added.
        """
        ...

    def on_remove_wall(self, i: int | slice, metadata: EnvironmentMetaData | list[EnvironmentMetaData]):
        """
        Called when a wall is removed from the environment. Should be overridden in subclasses.

        Args:
            i (int | slice): The index or slice of the walls that are being removed.
            metadata (EnvironmentMetaData | list[EnvironmentMetaData]): The metadata associated with
                the removed walls.
        """
        ...


class TransparentObjects(BaseCompilerCallback):
    """
    Callback class for handling transparent objects within a dynamic environment compilers.

    This class is designed to work with a compiler that has a visible plane capable of tracking transparency.
    It extends the BaseCompilerCallback with specific methods to handle the addition and removal of transparent objects.
    """
    def set_cache(self, cache: Mapping):
        self.requires = ['compiler']
        if not isinstance(cache['compiler'].visible_plane, LazyVisiblePlaneWithTransparancy):
            raise TypeError(f'Visible plane must be of type LazyVisiblePlaneWithTransparancy, got {type(cache["compiler"].visible_plane)} instead')
        super().set_cache(cache)

    def on_add_object(self, i: int | slice):
        """
        Tracks the addition of a transparent object or a range of transparent objects in the visible plane.

        Args:
            i (int | slice): The index or slice representing the object(s) to be tracked.
        """
        if isinstance(i, int):
            self.compiler.visible_plane.track_new(self.compiler.objects_metadata[i].vec_slice, True)
        elif isinstance(i, slice):
            for i_ in range(i.start, i.stop):
                self.compiler.visible_plane.track_new(self.compiler.objects_metadata[i_].vec_slice, True)

    def on_add_wall(self, i: int | slice):
        """
        Tracks the addition of a wall or a range of walls in the visible plane.

        Args:
            i (int | slice): The index or slice representing the wall(s) to be tracked.
        """
        if isinstance(i, int):
            self.compiler.visible_plane.track_new(self.compiler.objects_metadata[i].vec_slice, False)
        elif isinstance(i, slice):
            for i_ in range(i.start, i.stop):
                self.compiler.visible_plane.track_new(self.compiler.objects_metadata[i_].vec_slice, False)

    def on_remove(self, i: int | slice, metadata: EnvironmentMetaData | list[EnvironmentMetaData]):
        """
        Untracks objects that are removed from the visible plane.

        Args:
            i (int | slice): The index or slice representing the object(s) to be untracked.
            metadata (EnvironmentMetaData | list[EnvironmentMetaData]): The metadata of the object(s) being removed.
        """
        if isinstance(i, int):
            self.compiler.visible_plane.untrack(i)
        elif isinstance(i, slice):
            for i_ in range(i.start, i.stop):
                self.compiler.visible_plane.untrack(i_)
