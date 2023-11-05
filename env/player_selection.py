import numpy as np

class PlayerSelector:
    def __init__(self, player_names: list[str]):
        self.player_names = player_names
        self.index = 0
        self.current_player_name = self.player_names[self.index]
        self.direction = 1
        self.current_side = "left_team"

    def next(self):
        # TODO: verificar a lógica que seleciona o próximo player para jogar 
        # # alterna entre o time 1 e time 2 ?
        # # é do mesmo time até cada um dos 11 agir ?
        # # Lógica implementada abaixo: 
        # # 11 do time "left" jogam, depois 11 do time "right" jogam
        self.index = (self.index + 1) % len(self.player_names)
        self.current_player_name = self.player_names[self.index]
        if self.index == 11 or self.index == 0:
            self._toggle_side()
        return self.current_player_name

    def _toggle_side(self):
        self.direction = self.direction * -1
        #                      0        1             2
        #                     -3       -2            -1
        self.current_side = (None, "left_team", "right_team")[self.direction]
        # print("trocou team: ", self.direction, " | ", self.current_side)
