import sys
from typing import Any, Mapping


class Proxy:
    """
    A class that acts as a proxy for another object, allowing controlled access to its attributes.

    Attributes:
        obj (Any): The object being proxied.
    """
    def __init__(self, obj: Any, **kwargs):
        """
        Initializes the Proxy with the object and any additional attributes.

        Args:
            obj (Any): The object to be proxied.
            **kwargs: Additional attributes to be set on the Proxy.
        """
        self.__scope = list(kwargs.keys()) + [
            '__dict__',
            '__class__',
            '__setattr__',
            '__getattribute__',
            '__getattr__',
            'obj',
            'collect_magic',
            '_Proxy__scope'
        ]
        self.obj = obj

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def collect_magic(self):
        """
        Collects all magic methods from the proxied object that are not in the scope of the Proxy.

        Returns:
            dict: A dictionary of all the collected magic methods.
        """
        attributes = dir(self.obj)
        __dict__ = dict()
        for attr in attributes:
            if attr.startswith('__') and attr.endswith('__') and attr not in self.__scope:
                __dict__[attr] = getattr(self.obj, attr)
            try:
                __dict__[attr] = getattr(self.obj, attr)
            except:
                raise AttributeError(
                    f'Attribute {attr} cannot be accessed.\n'
                    f'The following error occurred:\n\n{sys.exc_info()[0]}'
                )
        return __dict__

    def __getattr__(self, __name: str) -> Any:
        """
        Overrides the default behavior for attribute access. If the attribute is not in the Proxy's scope,
        it returns the attribute from the proxied object. Otherwise, it returns the attribute from the Proxy.

        Args:
            __name (str): The name of the attribute.

        Returns:
            Any: The value of the attribute.
        """
        if __name not in self.__scope:
            return getattr(self.obj, __name)
        else:
            return self.__dict__[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Overrides the default behavior for setting an attribute. If the attribute is in the Proxy's scope,
        it sets the attribute on the Proxy. Otherwise, it sets the attribute on the proxied object.

        Args:
            __name (str): The name of the attribute.
            __value (Any): The value to set the attribute to.
        """
        if __name == '_Proxy__scope' or __name in self.__scope:
            super().__setattr__(__name, __value)
        else:
            setattr(self.obj, __name, __value)


class CallbacksCollection(list):
    """
    A collection of callback objects that extends the functionality of a standard list.
    This collection provides methods to execute a specified method on all callbacks,
    validate the requirements of each callback, and clean up unused cache entries.

    Methods:
        execute(method: str, *args, **kwargs):
            Executes a specified method on all callback objects in the collection.

        validate():
            Validates that all required cache entries for each callback are present.
            Raises a TypeError if a required cache entry is missing.

        clean_cache():
            Removes unused cache entries that are not required by any callback in the collection.
    """
    def execute(self, method: str, *args, **kwargs):
        """
        Executes a specified method on all callback objects in the collection.

        Args:
            method (str): The name of the method to execute on each callback object.
            *args: Variable length argument list to pass to the method.
            **kwargs: Arbitrary keyword arguments to pass to the method.

        Returns:
            tuple: A tuple containing the results of executing the method on each callback object.
        """
        return tuple([getattr(callback, method)(*args, **kwargs) for callback in self])

    def validate(self):
        """
        Validates that all required cache entries for each callback are present.
        Raises a TypeError if a required cache entry is missing.

        Raises:
            TypeError: If a required cache entry is missing for any callback in the collection.
        """
        if len(self):
            cache = self[0].cache
            for item in self:
                for request in item.requires:
                    if request not in cache:
                        raise TypeError(
                            f"Callback {item.__class__.__name__} requires {request} to be present in the cache."
                        )

    def clean_cache(self):
        """
        Removes unused cache entries that are not required by any callback in the collection.
        """
        if len(self):
            all_caches = self[0].cache.keys()
            used_caches = list()

            for item in self:
                used_caches += item.requires

            unused_caches = set(all_caches) - set(used_caches)

            for cache in unused_caches:
                del self[0].cache[cache]


class BaseCallback:
    """
    A base class for creating callback objects that can be used in any callbacks manager.
    """
    def __init__(self):
        """
        Initializes the BaseCallback instance with default values for cache and requires.
        """
        super().__init__()
        self._cache = None
        self._requires = list()

    @property
    def cache(self):
        """
        Returns the current cache.

        Returns:
            Mapping: The current cache.
        """
        return self._cache

    @property
    def requires(self):
        """
        Returns the current list of requirements to the cache.

        Returns:
            list: The current list of requirements.
        """
        return self._requires

    @requires.setter
    def requires(self, requires: list):
        """
        Sets the list of requirements.

        Args:
            requires (list): The new list of requirements.
        """
        self._requires = requires

    def set_cache(self, cache: Mapping):
        """
        Sets the cache with the provided mapping.

        Args:
            cache (Mapping): The new cache mapping.
        """
        self._cache = cache


class BaseCallbacksManager:
    """
    A base class for managing a collection of callbacks and a cache mapping.
    """
    def __init__(self, callbacks: list[BaseCallback] = None, cache: Mapping = None):
        """
        Initializes the BaseCallbacksManager with an optional cache.

        Args:
            cache (Mapping): An optional cache mapping to be used by the callbacks.
            callbacks (list[AbstractCallback]): An optional list of callbacks to be added to the manager.
        """
        super().__init__()
        self.callbacks = CallbacksCollection() if callbacks is None else CallbacksCollection(callbacks)
        self.cache = cache if cache is not None else dict()
        self.callbacks.execute('set_cache', self.cache)
        self.callbacks.validate()

    def add_callback(self, callback: Any):
        """
        Adds a new callback to the collection and validates its requirements.

        Args:
            callback (BaseCallback): The callback to be added to the simulation.
        """
        callback.set_cache(self.cache)
        self.callbacks.validate()
        self.callbacks.append(callback)

    def remove_callback(self, index: int):
        """
        Removes a callback from the collection by its index and cleans up the cache.

        Args:
            index (int): The index of the callback to be removed.
        """
        callback = self.callbacks.pop(index)
        callback.set_cache(None)
        self.callbacks.clean_cache()
