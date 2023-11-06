import numpy as np

from env.constants import PLAYER_COUNT
TOGGLE_X_ONLY = np.array([-1, 1])
TEAM_NAMES = ["right_team", "left_team"]

class PlayerSelector:
    def __init__(self, player_names: list[str]):
        self.player_names = player_names
        self._index = 0
        self._current_player_name = self.player_names[self._index]
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
        # TODO: verificar a lógica que seleciona o próximo player para jogar 
        # # alterna entre o time 1 e time 2 ?
        # # é do mesmo time até cada um dos 11 agir ?
        # # Lógica implementada abaixo:
        # # Apenas time "left" joga até acontecer kick off. 
        # # 11 do time "left" jogam, depois 11 do time "right" jogam
        if self._kickoff == True:
            self._index = (self._index + 1) % PLAYER_COUNT
        else:
            self._index = (self._index + 1) % (PLAYER_COUNT - 11)
            
        self._current_player_name = self.player_names[self._index]
        if self._index in (11, 0) and self._kickoff == True:
            self._toggle_side()
            print("Trocou time")
        return self._current_player_name

    def _toggle_side(self):
        self._direction = self._direction * TOGGLE_X_ONLY
        self._is_left_team ^= True # is_left_side XOR True
        self._is_left_team_str = TEAM_NAMES[+(self._is_left_team)]
