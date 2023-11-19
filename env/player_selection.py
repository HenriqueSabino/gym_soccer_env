import numpy as np

from env.constants import PLAYER_COUNT
TOGGLE_X_ONLY = np.array([-1, 1])
TEAM_NAMES = ["right_team", "left_team"]


class PlayerSelector:
    def __init__(self, player_names: list[str]):

        # Sort player names in the pós kickof order of turns
        self.player_order_to_play = [player_names[i//2] if i % 2 == 0 else player_names[i//2 + 11] for i in range(22)]

        self._index = 0
        self._current_player_name = self.player_order_to_play[self._index]
        self._x_foward_direction = np.array([1, 1])
        self._is_left_team = True
        self._is_left_team_str = "left_team"

        self.selector_logic_callback = self._before_kickoff_logic_callback


    def get_info(self) -> tuple[str, bool, np.array]:
        """
        Return needed info in step() of env
        """
        return (
            self._current_player_name, 
            self._x_foward_direction,
            self._is_left_team,
            self._is_left_team_str
        )


    def next_player(self, kickoff: bool) -> str:
        """
        Pass the turn to next player and updates the 
        internal _current_player_name, _is_left_team and _direction info.
        """
        self.selector_logic_callback()
               
        return self._current_player_name


    def _toggle_side(self):
        self._x_foward_direction = self._x_foward_direction * TOGGLE_X_ONLY
        self._is_left_team ^= True # is_left_side XOR True
        self._is_left_team_str = TEAM_NAMES[+(self._is_left_team)]
        print("Trocou time")


    def change_selector_logic(self):
        """
        The logic to select players change after kickoff
        """
        self.selector_logic_callback = self._after_kickoff_logic_callback
        self._index = 0

    
    def _before_kickoff_logic_callback(self):
        # Uncomment the line below if all players of the left team should play before kickoff
        # self_instance._index = (self_instance._index + 2) % PLAYER_COUNT
        self._current_player_name = self.player_order_to_play[self._index]


    def _after_kickoff_logic_callback(self):
        self._index = (self._index + 1) % PLAYER_COUNT
        self._current_player_name = self.player_order_to_play[self._index]
        self._toggle_side()

