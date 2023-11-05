from abc import ABC, abstractmethod

from env.player_action import PlayerAction

class ActionTranslator(ABC):

    @abstractmethod
    def translate_action(self, action, is_on_left_side: bool) -> PlayerAction:
        pass
