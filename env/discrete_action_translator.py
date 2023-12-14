from env.action_translator import ActionTranslator
from env.player_action import PlayerAction
import numpy as np

class DiscreteActionTranslator(ActionTranslator):
    
    @staticmethod
    def translate_action(action) -> PlayerAction:
        
        assert action[0] in list(range(8)), "Action must be in [0,...,11]"
        assert action[1] in list(range(5))
        action_to_direction = {
            0: np.array([1, 0]),  # Right (foward)
            1: np.array([-1, 0]), # Left (backward)
            2: np.array([0, 1]),  # Up
            3: np.array([0, -1]), # Down
            4: np.array([1, 1]),  # Right-Up (45° movement)
            5: np.array([-1, 1]), # Left-Up (45° movement)
            6: np.array([1, -1]), # Right-Down (45° movement)
            7: np.array([-1, -1]),# Left-Down (45° movement)
        }

        direction = DiscreteActionTranslator.__normalize_direction(action_to_direction[action[0]])

        # if not is_on_left_side:
        #     direction[1] *= -1

        playerAction = PlayerAction(direction)

        return playerAction
    
    @staticmethod
    def __normalize_direction(direction):
        if np.all(direction, where=lambda x: x == 0):
            return direction
        
        return direction / np.linalg.norm(direction)
        
