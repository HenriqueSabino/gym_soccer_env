from abc import ABC, abstractmethod

class ObservationBuilder(ABC):

    @abstractmethod
    def build_observation(self, left_team_positions: list, right_team_positions: list, ball_position: list, flip_side: bool):
        pass
