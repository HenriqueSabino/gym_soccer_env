from env.action_translator import ActionTranslator
from env.player_action import PlayerAction
import numpy as np

class DiscreteActionTranslator(ActionTranslator):
    
    @staticmethod
    def translate_action(action) -> PlayerAction:
        
        assert action in list(range(12)), "Action must be in [0,...,11]"
        
        action_to_direction = {
            0: np.array([1, 0]),  # Right (foward)
            1: np.array([-1, 0]), # Left (backward)
            2: np.array([0, 1]),  # Up
            3: np.array([0, -1]), # Down
            4: np.array([1, 1]),  # Right-Up (45° movement)
            5: np.array([-1, 1]), # Left-Up (45° movement)
            6: np.array([1, -1]), # Right-Down (45° movement)
            7: np.array([-1, -1]),# Left-Down (45° movement)
            8: np.array([0, 0]),  
            9: np.array([1, 0]),  
            10: np.array([1, 0]), 
            11: np.array([1, 0]), 
            12: np.array([0, 0]), 
        }

        direction = DiscreteActionTranslator.__normalize_direction(action_to_direction[action])

        # if not is_on_left_side:
        #     direction[1] *= -1
        if action == 8 and np.random.rand() < 0.5:
            direction = np.array([2, 2])
            print("Tentou roubar bola")
        elif action ==8:
            print("Falhou ao tentar roubar bola") 
        playerAction = PlayerAction(direction)

        return playerAction
    
    @staticmethod
    def __normalize_direction(direction):
        if np.all(direction, where=lambda x: x == 0):
            return direction
        
        return direction / np.linalg.norm(direction)
        
