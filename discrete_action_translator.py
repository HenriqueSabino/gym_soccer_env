from action_translator import ActionTranslator
from player_action import PlayerAction

class DiscreteActionTranslator(ActionTranslator):
    
    def translate_action(action) -> PlayerAction:
        return super().translate_action()