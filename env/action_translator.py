from abc import ABC, abstractmethod

from env.player_action import PlayerAction

class ActionTranslator(ABC):

    @abstractmethod
    def translate_action(action) -> PlayerAction:
        pass
