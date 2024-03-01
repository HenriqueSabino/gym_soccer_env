from typing import Union
import numpy as np
from gymnasium import ActionWrapper, spaces
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
    
class FlattenActionWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)

        current_action_space = env.action_space(agent="mock_string")
        
        assert isinstance(current_action_space, spaces.MultiDiscrete)

        self.dimensions = [dimension_size.n for dimension_size in np.array(current_action_space)]
        self.max_bound = np.prod(self.dimensions)
        
        self.flattened_size = 1
        for dimension_size in self.dimensions:
           self.flattened_size *= dimension_size


    def action_space(self, agent: str = None) -> spaces.Discrete:
        return spaces.Discrete(self.flattened_size)


    # copy this to make a gymnasium wrapper if nedeed
    # def action(self, flattened_action_index: int) -> spaces.MultiDiscrete:
    #     # print("[FlattenActionWrapper] Recebeu: ", flattened_action_index, type(flattened_action_index))

    #     if isinstance(flattened_action_index, int) or \
    #        isinstance(flattened_action_index, np.int64):
    #         assert 0 <= flattened_action_index < self.max_bound, "Invalid action index"

    #         # convert to MultiDiscrete action
    #         multi_discrete_action_indexes = np.unravel_index(
    #             flattened_action_index, 
    #             self.dimensions
    #         )
    #     elif isinstance(flattened_action_index, list) or \
    #          isinstance(flattened_action_index, np.ndarray):
            
    #         for a in flattened_action_index:
    #             assert 0 <= a < self.max_bound, "Invalid action index"

    #             # convert to MultiDiscrete action
    #             multi_discrete_action_indexes = []
    #             multi_discrete_action_indexes.append(np.unravel_index(
    #                 a, 
    #                 self.dimensions
    #             ))
    #     else:
    #         raise TypeError(f"Argument must be type int or list[int]. Got type {type(flattened_action_index)}")

        
    #     return multi_discrete_action_indexes
    

    def step(self, action: Union[int, list[int]]) -> None:
        # print("[FlattenActionWrapper] Recebeu: ", action, type(action))

        if isinstance(action, int) or \
           isinstance(action, np.int64):
            assert 0 <= action < self.max_bound, "Invalid action index"

            # convert to MultiDiscrete action
            multi_discrete_action_indexes = np.unravel_index(
                action, 
                self.dimensions
            )
        elif isinstance(action, list) or \
             isinstance(action, np.ndarray):
            
            for a in action:
                assert 0 <= a < self.max_bound, "Invalid action index"

                # convert to MultiDiscrete action
                multi_discrete_action_indexes = []
                multi_discrete_action_indexes.append(np.unravel_index(
                    a, 
                    self.dimensions
                ))
        else:
            raise TypeError(f"Argument must be type int or list[int]. Got type {type(action)}")

        self.env.step(multi_discrete_action_indexes)


    def __str__(self) -> str:
        return str(self.env)
