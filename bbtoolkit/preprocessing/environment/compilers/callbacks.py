from bbtoolkit.preprocessing.environment.compilers import BaseCompilerCallback, DynamicEnvironmentCompiler, EnvironmentMetaData
from bbtoolkit.preprocessing.environment.visible_planes import LazyVisiblePlaneWithTransparancy


class TransparentObjects(BaseCompilerCallback):
    """
    Callback class for handling transparent objects within a dynamic environment compilers.

    This class is designed to work with a compiler that has a visible plane capable of tracking transparency.
    It extends the BaseCompilerCallback with specific methods to handle the addition and removal of transparent objects.
    """
    def set_compiler(self, compiler: DynamicEnvironmentCompiler):
        """
        Sets the compiler instance for this callback and checks if the visible plane is of the correct type.

        Args:
            compiler (DynamicEnvironmentCompiler): The compiler instance to be used with this callback.

        Raises:
            TypeError: If the compiler's visible plane is not an instance of LazyVisiblePlaneWithTransparancy.
        """
        if not isinstance(compiler.visible_plane, LazyVisiblePlaneWithTransparancy):
            raise TypeError(f'Visible plane must be of type LazyVisiblePlaneWithTransparancy, got {type(compiler.visible_plane)} instead')
        super().set_compiler(compiler)

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