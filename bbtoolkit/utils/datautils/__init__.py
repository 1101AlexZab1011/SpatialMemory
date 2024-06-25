from collections import OrderedDict
from copy import deepcopy
import hashlib
import logging
import pickle
import sys
from typing import Any, Callable
import os
from abc import ABC, abstractmethod
import concurrent.futures
import functools


def read_pkl(path: str) -> Any:
    """
    Read and deserialize an object from a pickle file.

    Args:
        path (str): The path to the pickle file.

    Returns:
        Any: The deserialized object.

    """
    with open(
        path,
        'rb'
    ) as file:
        content = pickle.load(
            file
        )
    return content


def save_pkl(content: Any, path: str | os.PathLike):
    """
    Save an object using pickle serialization.

    Args:
    content (Any): the object to be serialized and saved.
    path (str | os.PathLike): the path and filename where the serialized object will be saved.
    The file extension must be '.pkl'.

    Raises:
    OSError: if the file extension of path is not '.pkl'.

    Returns:
    None
    """

    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')

    pickle.dump(content, open(path, 'wb'))


class AbstractWritable(ABC):
    """
    An abstract base class for objects that can be saved and loaded.

    This class defines two abstract methods, 'save' and 'load', which should be
    implemented by concrete subclasses.
    """

    @abstractmethod
    def save(self, filename):
        """
        Save the object to a file.

        Args:
            filename (str): The name of the file to which the object should be saved.

        Raises:
            NotImplementedError: This method must be implemented in concrete subclasses.
        """
        pass

    @staticmethod
    @abstractmethod
    def load(filename):
        """
        Load an object from a file.

        Args:
            filename (str): The name of the file from which the object should be loaded.

        Returns:
            Writable: An instance of a concrete subclass loaded from the file.

        Raises:
            NotImplementedError: This method must be implemented in concrete subclasses.
        """
        pass


class WritablePickle(AbstractWritable):
    def save(self, savepath: str):
        """
        Save the object to a .pkl file.

        Args:
            savepath (str): The path to the file where the object will be saved in .pkl format.
        """
        save_pkl(self, savepath)

    @staticmethod
    def load(path: str):
        """
        Load an object from a specified file path using pickle deserialization.

        Args:
            path (str): The file path from which to load the object.

        Returns:
            object: The loaded object.
        """
        return read_pkl(path)


class Copyable(ABC):
    """
    An abstract base class for copyable objects
    """
    def copy(self):
        """
        Create a deep copy of the object.
        """
        return deepcopy(self)


class Cached:
    """
    A class for caching the results of another function with a specified maximum size for the cache.

    Args:
        max_size (int): The maximum size of the cache. If the cache exceeds this size, the oldest entry is evicted.
        cache_storage (OrderedDict): The storage for the cache. If None, a new OrderedDict is created.

    """
    def __init__(self, max_size: int = 100, cache_storage: OrderedDict = None):
        self.max_size = max_size
        self.cache = cache_storage if cache_storage is not None else OrderedDict()

    def encrypt_key(self, func: Callable, *args, **kwargs):
        """
        Helper method to create a unique hash for the parameters.

        Args:
            func (Callable): The function to be cached.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            str: The MD5 hash of the concatenated string representation of args and kwargs.
        """
        all_kwargs = {**dict(zip(func.__code__.co_varnames, args)), **kwargs}
        key = hashlib.md5(str(all_kwargs).encode()).hexdigest()
        return key

    def __call__(self, func: Callable):
        def wrapper(*args, **kwargs):
            """
            Wrapper function that adds caching functionality to the decorated function.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Any: The result of the decorated function.
            """
            key = self.encrypt_key(func, *args, **kwargs)

            # Check if the result is already in the cache
            if key in self.cache:
                # Move the key to the end to mark it as the most recently used
                self.cache.move_to_end(key)
                return self.cache[key]

            # Call the function if the result is not in the cache
            result = func(*args, **kwargs)

            # Add the result to the cache
            self.cache[key] = result

            # Check and evict the oldest entry if the cache size exceeds the maximum
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            return self.cache[key]

        return wrapper


def asynchronous(func):
    """
    A decorator that makes a function execute asynchronously.

    This decorator wraps a function and makes it execute in a separate thread.
    The wrapped function is submitted to a ThreadPoolExecutor and its future is returned.

    Args:
        func (callable): The function to be executed asynchronously.

    Returns:
        concurrent.futures.Future: A Future object representing the execution of the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function that is called instead of 'func'.

        This function creates a ThreadPoolExecutor, submits 'func' to it for execution,
        and returns the resulting Future object.

        Args:
            *args: Variable length argument list to be passed to 'func'.
            **kwargs: Arbitrary keyword arguments to be passed to 'func'.

        Returns:
            concurrent.futures.Future: A Future object representing the execution of 'func'.
        """
        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(func, *args, **kwargs)
        return future
    return wrapper


def is_custom_class(cls):
    """
    Checks if the given class is a custom (user-defined) class.

    This function determines if a class is custom based on its module. Classes defined in the '__main__'
    module or in modules not part of the standard library or third-party packages are considered custom.

    Args:
        cls (type): The class to check.

    Returns:
        bool: True if the class is custom, False otherwise.
    """
    if not isinstance(cls, type):
        raise ValueError("Provided argument is not a class.")

    # Check if the class is defined in the '__main__' module
    if cls.__module__ == '__main__':
        return True

    # Check if the class is a built-in class
    if cls.__module__ == 'builtins':
        return False

    # Attempt to determine if the class is defined in a standard library or third-party module
    try:
        module = __import__(cls.__module__)
        # Check if the module is a built-in module or if its file attribute points to a site-packages directory
        if (hasattr(module, '__file__') and module.__file__ is not None and 'site-packages' in module.__file__) or (cls.__module__ in sys.builtin_module_names):
            return False
    except ImportError:
        # If the module cannot be imported, it might be a user-defined module
        return True

    return True


def ismutable(obj: Any) -> bool:
    """
    Checks if the given object is mutable.

    This function determines if an object is mutable based on its type and behavior. It first checks if the object
    is an instance of immutable types (int, float, bool, str, tuple, frozenset, bytes, complex). If not, it further
    checks if the object supports the addition operation (`__add__`) and attempts to modify the object by adding it
    to itself. The object is considered mutable if it can be successfully modified in this way.

    Args:
        obj (Any): The object to check for mutability.

    Returns:
        bool: True if the object is mutable, False otherwise.
    """
    # try:
    #     obj = deepcopy(obj)
    # except:
    #     print(type(obj))
    #     raise

    if isinstance(obj, (int, float, bool, str, tuple, frozenset, bytes, complex)):
        return False

    # if hasattr(obj, '__add__'):
    #     try:
    #         obj_ = obj
    #         obj_ += obj
    #         return id(obj_) == id(obj)
    #     except:
    #         return True

    # return True

    mutable_methods = [
        '__iadd__', '__isub__', '__imul__', '__itruediv__', '__ifloordiv__', '__imod__',
        '__ipow__', '__ilshift__', '__irshift__', '__iand__', '__ixor__', '__ior__',
        '__setitem__', '__delitem__', 'append', 'extend', 'pop', 'remove', 'clear',
        'update', 'popitem', 'add', 'discard'
    ]

    return any(hasattr(obj, method) for method in mutable_methods)


def remove_files_from_dir(directory):
    """
    Removes all files from the specified directory.

    Args:
        directory (str): The path to the directory from which files will be removed.

    Returns:
    None
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                logging.info(f"Removed: {file_path}")
            else:
                logging.info(f"Skipped: {file_path} (Not a file)")
        except Exception as e:
            logging.error(f"Failed to remove {file_path}. Reason: {e}")
