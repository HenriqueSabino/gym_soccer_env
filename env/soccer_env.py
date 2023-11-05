import random
from typing import Any, SupportsFloat
from gymnasium.core import Env, RenderFrame
from gymnasium import spaces 
from env.discrete_action_translator import DiscreteActionTranslator
from env.player_selection import PlayerSelector
from env.field_drawer import FieldDrawer
import numpy as np

from env.constants import FIELD_WIDTH, FIELD_HEIGHT, PLAYER_VELOCITY, POINTS


class SoccerEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, render_mode="humam", action_format='discrete', render_scale=8) -> None:
        super().__init__()

        self.render_mode = render_mode

        self.field_width = FIELD_WIDTH
        self.field_height = FIELD_HEIGHT
        self.field_drawer = FieldDrawer(render_scale, border_size=2)
        
        self.__initialize_players()
        self.__initialize_action_translator(action_format)
        self.__initialize_render_function(render_mode)
        self.player_selector = PlayerSelector(self.player_names)

        # TODO: verificar a definição de observaiton space e action space
        # # precisa conter o domínio (conjunto) de valores possíveis da observação
        # # Se uma observação não estivar dentro desse domínio, então vai sair um warning no terminal. 
        # # Não vai dar erro e tudo funciona normalmente.
        # # References to see __init__():
        # # https://pettingzoo.farama.org/content/environment_creation/
        # # https://www.gymlibrary.dev/content/environment_creation/
        # # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict({
            # "image": None, # Parte do AlphaZero
            "left_team": spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
                dtype=np.float32,
            ),
            "right_team": spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
                dtype=np.float32,
            ),
            "ball_position": spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
                dtype=np.float32,
            )
        })


    def render(self, ) -> RenderFrame | list[RenderFrame] | None:
        return self.render_function()
    

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.observation = {
            # "image": None, # TODO: Adaptar pro AlphaZero usando planes (2Darray de 0 e 1)
            "left_team": self.all_start_positions[:11],
            "right_team": self.all_start_positions[11:],
            "ball_position": np.array(POINTS["center"], dtype=np.float32),
        }

        self.rewards = {agent: 0 for agent in self.player_names}
        self._cumulative_rewards = {agent: 0 for agent in self.player_names}
        self.terminations = {agent: False for agent in self.player_names}
        self.truncations = {agent: False for agent in self.player_names}
        self.infos = {agent: {} for agent in self.player_names}
        self.num_moves = 0

        # TODO: arrumar reset pra deixar o código limpo
        # # Muita coisa do reset ta no __init__
        # # References to see reset() method in:
        # # https://pettingzoo.farama.org/content/environment_creation/
        # # https://www.gymlibrary.dev/content/environment_creation/
        # # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
    

    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # assert len(action) == 22
        
        t_action = self.action_translator(action)
        direction = self.player_selector.direction
        team = self.player_selector.current_side
        player_name = self.player_selector.current_player_name
        
        obs_index = self.player_names_mapping[player_name]
        old_position = self.observation[team][obs_index]
        # TODO: Verificar se está correto direction ficar -1 quando acion for 'up' ou 'down'
        new_position = old_position + t_action * PLAYER_VELOCITY * direction
        new_position = np.clip(
            new_position, 
            self.observation_space[team].low, 
            self.observation_space[team].high
        )
        self.observation[team][obs_index] = new_position
        
        print(team, player_name, f"move from ({old_position}) to ({new_position})", f"Indexes({obs_index}, {self.player_selector.index})")
        
        # Passa a vez pro próximo jogador atualizando player, team e direction
        self.player_selector.next()

        #TODO: Implement how action alters the game state
        # # ta super bem explicado no link environment creation da farama
        # # Ver o step() dos exemplos abaixo:
        # # acho que precisa retornar reward, terminated, truncation, info
        # # return self.observation, reward, terminated, None, {}
        # # https://pettingzoo.farama.org/content/environment_creation/
        # # https://www.gymlibrary.dev/content/environment_creation/
        # # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
        
        # return super().step(action)
        return self.observation, 0, False, None, {}
    

    def __initialize_players(self, num_agents = 22):

        def random_coordinates_generator(n: int = 22, gen_int: bool = True):
            if gen_int:
                x = np.random.randint(0, FIELD_WIDTH+1, size=n)
                y = np.random.randint(0, FIELD_HEIGHT+1, size=n)
            else:
                x = np.random.uniform(0, FIELD_WIDTH, size=n)
                y = np.random.uniform(0, FIELD_HEIGHT, size=n)

            return np.column_stack((x, y))
        
        # First 11 players will be left side players and last 11 will be right side
        self.all_start_positions = random_coordinates_generator() # posições de todos os jogadores
        self.player_names = ["player_" + str(r) for r in range(num_agents)]
        self.player_names_mapping = dict( # mapping agent_name to index of observation channel
            zip(self.player_names, list(range(11)) * 2)
        )


    def __initialize_action_translator(self, action_format):
        if action_format == 'discrete':
            self.action_translator = DiscreteActionTranslator.translate_action
        else:
            raise Exception("Action format is not recognized")
        

    def __initialize_render_function(self, render_mode: str = "humam") -> None:

        def human_render():
            array_1 = self.observation["left_team"]
            array_2 = self.observation["right_team"]
            all_positions = np.concatenate([array_1, array_2], axis=0)
            field_image = self.field_drawer.draw_field(all_positions)
            return field_image
        
        def rgb_array_render():
            pass # TODO: completar essa função pensando em como o AlphaZero usaria

        render_functions = {
            "human": human_render,
            "rgb_array": rgb_array_render 
        }
        self.render_function = render_functions[render_mode]
