import numpy as np
from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME

# Not used. Can be used in with a refactor in soccer_env.py
# class Player:
#     name: str
#     position: np.array
#     direction: np.array # (+-x, 1)
#     is_defending: bool
#     is_left_side: bool

#     def __init__(self, 
#                  name: str, 
#                  position: np.array, 
#                  direction: np.array, 
#                  team_name: str
#                 ) -> None:
        
#         self.position = position
#         self.direction = direction
#         self.is_defending = False
#         self.team_name = team_name
#         self.is_left_side = True if team_name == TEAM_LEFT_NAME else False
