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
        self._direction = np.array([1, 1])
        self._is_left_team = True
        self._is_left_team_str = "left_team"
        self._kickoff = False


    def get_info(self) -> tuple[str, bool, np.array]:
        """
        Return needed info in step() of env
        """
        return (
            self._current_player_name, 
            self._direction,
            self._is_left_team,
            self._is_left_team_str
        )


    def next_(self) -> str:
        """
        Pass the turn to next player and updates the 
        internal _current_player_name, _is_left_team and _direction info.
        """
        if self._kickoff == True:
            self._index = (self._index + 1) % PLAYER_COUNT
            self._current_player_name = self.player_order_to_play[self._index]
            self._toggle_side()
        else:
            # if all player of left team should play before kickoff, then uncomment this line 
            # self._index = (self._index + 2) % PLAYER_COUNT

            # First left team player always play before kickof
            self._current_player_name = self.player_order_to_play[self._index]
               
        return self._current_player_name

    def _toggle_side(self):
        self._direction = self._direction * TOGGLE_X_ONLY
        self._is_left_team ^= True # is_left_side XOR True
        self._is_left_team_str = TEAM_NAMES[+(self._is_left_team)]
        print("Trocou time")
