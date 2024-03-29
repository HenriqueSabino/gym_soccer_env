from env.modules.action.abstract_action_translator import ActionTranslator
from env.modules.action.player_action import PlayerAction
import numpy as np

from gymnasium import spaces

class DiscreteActionTranslator(ActionTranslator):
    
    def translate_action(self, action) -> PlayerAction:
        
        assert action[0] in list(range(9)), "Direction must be in [0,...,8]"
        assert action[1] in list(range(5)), "Action must be in [0,...,4]"

        action_to_direction = {
            0: np.array([1, 0]),    # Right (foward)
            1: np.array([-1, 0]),   # Left (backward)
            2: np.array([0, -1]),   # Up
            3: np.array([0, 1]),    # Down
            4: np.array([1, -1]),   # Right-Up (45° movement)
            5: np.array([-1, -1]),  # Left-Up (45° movement)
            6: np.array([1, 1]),    # Right-Down (45° movement)
            7: np.array([-1, 1]),   # Left-Down (45° movement)
            8: np.array([0, 0]),    # No direction
        }

        direction = DiscreteActionTranslator.__normalize_direction(action_to_direction[action[0]])
        
        playerAction = PlayerAction(direction)

        return playerAction
    
    def action_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([9, 5])
    
    @staticmethod
    def __normalize_direction(direction):
        if np.all(direction == 0):
            return direction
        
        return direction / np.linalg.norm(direction)
        
