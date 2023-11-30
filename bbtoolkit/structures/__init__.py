import sys
from typing import Any


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