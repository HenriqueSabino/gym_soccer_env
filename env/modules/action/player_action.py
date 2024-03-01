import numpy as np

class PlayerAction:
    
    def __init__(self, direction: np.ndarray) -> None:
        self._direction = direction

    @property
    def direction(self) -> np.ndarray:
        return self._direction
    
    @direction.setter
    def direction(self, direction: np.ndarray):
        self._direction = direction
