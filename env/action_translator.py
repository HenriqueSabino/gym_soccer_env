from abc import ABC, abstractmethod

from env.player_action import PlayerAction

class ActionTranslator(ABC):

    @abstractmethod
    def translate_action(self, action) -> PlayerAction:
        pass

    @abstractmethod
    def action_space(self):
        pass
