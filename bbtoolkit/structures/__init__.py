from typing import Any, Literal, Mapping
from bbtoolkit.utils.datautils import is_custom_class, ismutable


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

        # self.obj = obj
        super().__setattr__('obj', obj)
        for key, value in kwargs.items():
            super().__setattr__(key, value)

    def __getattribute__(self, __name: str) -> Any:
        """
        Overrides the default behavior for attribute access. If the attribute is not in the Proxy's scope,
        it returns the attribute from the proxied object. Otherwise, it returns the attribute from the Proxy.

        Args:
            __name (str): The name of the attribute.

        Returns:
            Any: The value of the attribute.
        """
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return super().__getattribute__('obj').__getattribute__(__name)

    def __setattribute__(self, __name: str, __value: Any) -> None:
        """
        Overrides the default behavior for setting an attribute. If the attribute is in the Proxy's scope,
        it sets the attribute on the Proxy. Otherwise, it sets the attribute on the proxied object.

        Args:
            __name (str): The name of the attribute.
            __value (Any): The value to set the attribute to.
        """
        super().__setattr__(__name, __value)

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Overrides the default behavior for setting an attribute. If the attribute is in the Proxy's scope,
        it sets the attribute on the Proxy. Otherwise, it sets the attribute on the proxied object.

        Args:
            __name (str): The name of the attribute.
            __value (Any): The value to set the attribute to.
        """
        if hasattr(self, __name):
            super().__setattr__(__name, __value)
        else:
            self.obj.__setattr__(__name, __value)


class DotDict(dict):
    """
    A dictionary subclass that supports accessing and setting items via attribute notation (dot notation) in addition to the standard item notation (square brackets).

    Attributes:
        No public attributes.
    """

    def __getattr__(self, item):
        """
        Allows attribute-style access (dot notation) to dictionary items.

        Args:
            item (str): The key whose value is to be returned.

        Returns:
            The value associated with 'item'.

        Raises:
            AttributeError: If the item is not found in the dictionary.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Allows setting dictionary items using attribute-style assignment (dot notation).

        Args:
            key (str): The key to which the value is to be assigned.
            value: The value to be assigned to the key.
        """
        self[key] = value

    def __delattr__(self, item):
        """
        Allows deleting dictionary items using attribute-style notation (dot notation).

        Args:
            item (str): The key to be deleted.

        Raises:
            AttributeError: If the item is not found in the dictionary.
        """
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __repr__(self) -> str:
        """
        Returns a string representation of the DotDict.
        """
        return super().__repr__()


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

        for callback in self:
            callback._set_cache_attrs(on_repeat='ignore')

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

    def set_cache(self, cache: Mapping, on_repeat: Literal['raise', 'ignore', 'overwrite'] = 'raise'):
        """
        Sets the cache with the provided mapping.

        Args:
            cache (Mapping): The new cache mapping.
            on_repeat (Literal['raise', 'ignore', 'overwrite']): The behavior when a cache key is already an attribute.
        """
        self._cache = cache
        self._set_cache_attrs(on_repeat=on_repeat)

    def _set_cache_attrs(self, on_repeat: Literal['raise', 'ignore', 'overwrite'] = 'raise'):
        """
        Sets the cache attributes for the callback.

        Args:
            on_repeat (Literal['raise', 'ignore', 'overwrite']): The behavior when a cache key is already an attribute.
        """
        for key in self.requires:
            if key not in self.__dict__:
                if key not in self.cache:
                    raise KeyError(
                        f'Cache is missing a required key {key} for {self.__class__.__name__}.\n'
                        'Please add the required key to the cache.'
                    )
                if not ismutable(self.cache[key]) and not is_custom_class(self.cache[key].__class__):
                    raise ValueError(
                        f'Cache has an immutable value of type {type(self.cache[key])} under the key {key}.\n'
                        'Only mutable values are allowed for caching.'
                    )
                setattr(self, key, self.cache[key])
            else:
                match on_repeat:
                    case 'raise':
                        raise AttributeError(
                            f'Callback {self.__class__.__name__} already has an attribute named {key}.\n'
                            'Please rename the cache key or the attribute.'
                        )
                    case 'ignore':
                        pass
                    case 'overwrite':
                        setattr(self, key, self.cache[key])
                    case _:
                        raise ValueError(
                            f'Invalid value for on_repeat: {on_repeat}.\n'
                            'Valid values are "raise", "ignore", and "overwrite".'
                        )

    def __getattribute__(self, __name: str) -> Any:
        '''
        Overrides the default behavior for attribute access by checking if the attribute is in the cache and mutable.

        Args:
            __name (str): The name of the attribute to access.

        Returns:
            Any: The value of the attribute.

        Notes:
            * If the attribute is in the cache, it checks if the value is the same as the one in the cache. All values must be mutable.
        '''
        out = super().__getattribute__(__name)
        if __name not in ('_cache', 'cache', '_requires', '__dict__'):
            if self.cache is not None and __name in self.cache and id(self.cache[__name]) != id(out):
                    raise AttributeError(
                        f'Attribute {__name} of {self.__class__.__name__} is not the same as the value in the cache.\n'
                        f'Check for non-in-place operations happening to {__name}.'
                    )
        return out


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

    def __add__(self, other: 'BaseCallbacksManager') -> 'BaseCallbacksManager':
        """
        Merges two callbacks managers by combining their caches and callbacks.

        Args:
            other (BaseCallbacksManager): The other callbacks manager to merge with.

        Returns:
            BaseCallbacksManager: The merged callbacks manager.
        """
        self.cache.update(other.cache)
        new_callbacks = self.callbacks + other.callbacks
        new_manager = BaseCallbacksManager(new_callbacks, self.cache)
        new_manager.callbacks.clean_cache()
        new_manager.callbacks.validate()
        return new_manager
