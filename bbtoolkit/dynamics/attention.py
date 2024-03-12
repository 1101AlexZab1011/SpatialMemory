from abc import ABC, abstractmethod
import numpy as np


class AbstractAttention(ABC):
    """
    An abstract class that defines the basic structure for implementing attention mechanisms.

    Methods:
        __call__(objects: list[np.ndarray], return_index: bool = False) -> np.ndarray:
            Abstract method that must be implemented by subclasses. It defines how the attention mechanism
            operates on a list of objects.
    """
    @abstractmethod
    def __call__(self, objects: list[np.ndarray], return_index: bool = False) -> np.ndarray:
        pass


class RhythmicAttention(AbstractAttention):
    """
    An implementation of the AbstractAttention class that selects objects to pay attention to based on a rhythmic pattern.

    Attributes:
        freq (float): The frequency of the attention cycle.
        dt (float): The time step between updates.
        n_objects (int): The number of objects to consider for attention.
        cycle (int): The number of time steps in one complete attention cycle.
        timer (int): A counter used to determine the current position within the attention cycle.
        attend_to (int or None): The index of the currently attended object. None if no object is being attended.
        attention_priority (np.ndarray): An array indicating the priority of each object for receiving attention.

    Methods:
        step() -> bool:
            Advances the timer and determines if the attention cycle is complete.
        visible_objects(objects: list[np.ndarray]) -> list[int]:
            Determines which objects are visible (i.e., have a non-zero size).
        __call__(objects: list[np.ndarray], return_index: bool = False) -> np.ndarray:
            Processes a list of objects and determines which object to pay attention to based on the rhythmic pattern.
    """
    def __init__(self, freq: float, dt: float, n_objects: int):
        """
        Initializes the RhythmicAttention object with the specified frequency, time step, and number of objects.

        Args:
            freq (float): The frequency of the attention cycle.
            dt (float): The time step between updates.
            n_objects (int): The number of objects to consider for attention.
        """
        self.cycle = int(1/(freq * dt))
        self.timer = 0
        self.n_objects = n_objects
        self.attend_to = None
        self.attention_priority = np.zeros(n_objects)

    def step(self) -> bool:
        """
        Advances the timer and determines if the attention cycle is complete.

        Returns:
            bool: True if the cycle is complete, False otherwise.
        """
        self.timer += 1
        self.timer %= self.cycle
        return self.timer == 0

    @staticmethod
    def visible_objects(objects: list[np.ndarray]) -> list[int]:
        """
        Determines which objects are visible (i.e., have a non-zero size).

        Args:
            objects (list[np.ndarray]): A list of objects represented as numpy arrays.

        Returns:
            list[int]: A list indicating the visibility of each object (True for visible, False for not visible).
        """
        return np.array([arr.size > 0 for arr in objects])

    def __call__(self, objects: list[np.ndarray], return_index: bool = False) -> np.ndarray:
        """
        Processes a list of objects and determines which object to pay attention to based on the rhythmic pattern.

        Args:
            objects (list[np.ndarray]): A list of objects represented as numpy arrays.
            return_index (bool, optional): If True, returns the index of the attended object instead of the object itself.
                                           Defaults to False.

        Returns:
            np.ndarray: The attended object or its index, depending on the value of `return_index`.
        """
        # True for all objects that are in the field of view in the current moment
        visible_objects = self.visible_objects(objects)

        single_visible_object = False
        no_visible_objects = False

        # If there are no objects to pay attention to, pay attention to the first visible object or do not pay attention to any object
        if self.attend_to is None:
            visible_objects_indices = np.where(visible_objects)[0]

            if visible_objects_indices.size:
                self.attend_to = visible_objects_indices[0]

        # Increase the priority of the objects that are visible, zero the priority of the attended object and invisible objects
        self.attention_priority = self.attention_priority*visible_objects + visible_objects

        no_visible_objects = np.all(np.logical_not(self.attention_priority))
        single_visible_object = True if visible_objects.sum() == 1 and self.attend_to == np.argmax(self.attention_priority) else False

        self.attention_priority[self.attend_to] = 0

        if self.step(): # time to switch attention
            if not single_visible_object: # if there is only one visible object, do not switch attention
                self.attend_to = np.argmax(self.attention_priority) if not no_visible_objects else None # if there are no visible objects, do not attend to any object

        # Return the object that is currently being attended to
        if return_index:
            return self.attend_to
        else:
            return objects[self.attend_to] if self.attend_to is not None else np.array([])