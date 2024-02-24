import numpy as np
from gymnasium import ActionWrapper, spaces
    
class FlattenActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        assert isinstance(env.action_space, spaces.MultiDiscrete)

        self.dimensions = [dimension_size.n for dimension_size in np.array(env.action_space)]
        flattened_size = 1
        for dimension_size in self.dimensions:
           flattened_size *= dimension_size

        self.action_space = spaces.Discrete(flattened_size)

    def action(self, flattened_action_index: int) -> spaces.MultiDiscrete:

        assert 0 <= flattened_action_index < np.prod(self.dimensions), "Invalid action index"

        # convert to MultiDiscrete action
        multi_discrete_action_indexes = np.unravel_index(
            flattened_action_index, 
            self.dimensions
        )
        
        return multi_discrete_action_indexes
