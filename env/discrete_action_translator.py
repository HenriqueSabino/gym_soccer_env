from env.action_translator import ActionTranslator
from env.player_action import PlayerAction
import numpy as np

class DiscreteActionTranslator(ActionTranslator):
    
    
    @staticmethod
    def translate_action(action) -> PlayerAction:
        
        assert action in list(range(8)), "Action must be in [0,...,7]"

        action_to_direction = {
            0: np.array([1, 0]),  # Right (foward)
            1: np.array([-1, 0]), # Left (backward)
            2: np.array([0, 1]),  # Up
            3: np.array([0, -1]), # Down
            4: np.array([1, 1]),  # Right-Up (45° movement)
            5: np.array([-1, 1]), # Left-Up (45° movement)
            6: np.array([1, -1]), # Right-Down (45° movement)
            7: np.array([-1, -1]),# Left-Down (45° movement)
            8: None,  # To be implemented in another week
            9: None,  # To be implemented in another week
            10: None, # To be implemented in another week
            11: None, # To be implemented in another week
            12: None, # To be implemented in another week
        }

        # TODO: Faltar implementar e retornar PlayerAction em vez de retornar um numpy array

        # return super().translate_action() pode apagar essa linha ? precisa chamer o super ?
        return action_to_direction[action]
        
