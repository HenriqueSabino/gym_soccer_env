from typing import Union
from typing import Any, SupportsFloat
from gymnasium.core import Env, RenderFrame
from gymnasium import spaces
from env.discrete_action_translator import DiscreteActionTranslator
from env.player_selection import PlayerSelector
from env.field_drawer import FieldDrawer
from env.image_observation_builder import ImageObservationBuilder
from env.dict_observation_builder import DictObservationBuilder 
import numpy as np

from env.constants import FIELD_WIDTH, FIELD_HEIGHT, PLAYER_VELOCITY, POINTS


class SoccerEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4
    }

    def __init__(self, 
                 render_mode="rgb_array", 
                 action_format='discrete', 
                 observation_format='image', 
                 render_scale=8
                ) -> None:
        super().__init__()

        self.render_mode = render_mode

        self.field_width = FIELD_WIDTH
        self.field_height = FIELD_HEIGHT
        self.field_drawer = FieldDrawer(render_scale, border_size=2)
        
        self.__initialize_players()
        self.__initialize_action_translator(action_format)
        self.__initialize_render_function(render_mode)
        self.__initialize_observation_builder(observation_format)
        self.observation_type = observation_format

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
        self.observation_space = self._get_observation_space()


    def _get_observation_space(self):
        if self.observation_type == 'image':
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.field_height, self.field_width, 3),
                dtype=np.uint8
            )
        elif self.observation_type == 'dict':
            return spaces.Dict({
                "left_team": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([FIELD_HEIGHT, FIELD_WIDTH], dtype=np.float32),
                    dtype=np.float32,
                ),
                "right_team": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([FIELD_HEIGHT, FIELD_WIDTH], dtype=np.float32),
                    dtype=np.float32,
                ),
                "ball_position": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([FIELD_HEIGHT, FIELD_WIDTH], dtype=np.float32),
                    dtype=np.float32,
                )
            })
        else:
            raise ValueError("Invalid observation_type. Choose 'image' or 'dict'.")


    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        return self.render_function()
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # If needed -> References to see reset() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226

        self.observation = self.observation_builder.build_observation(
            self.all_start_positions[:11],
            self.all_start_positions[11:],
            np.array(POINTS["center"], dtype=np.float32)
        )

        self.rewards = {agent: 0 for agent in self.player_names}
        self._cumulative_rewards = {agent: 0 for agent in self.player_names}
        self.terminations = {agent: False for agent in self.player_names}
        self.truncations = {agent: False for agent in self.player_names}
        self.infos = {agent: {} for agent in self.player_names}

        return self.observation, self.infos


    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        # If needed -> References to see step() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
        
        # assert len(action) == 22

        t_action = self.action_translator(action)
        
        player_name, direction, _, team  = self.player_selector.get_info()
        obs_index = self.player_name_to_obs_index[player_name]
        old_position = self.observation[team][obs_index]
        new_position = old_position + t_action.direction * PLAYER_VELOCITY * direction
        new_position = np.clip(
            new_position, 
            self.observation_space["left_team"].low, 
            self.observation_space["left_team"].high
        )
        self.observation[team][obs_index] = new_position
        
        print(team, player_name, f"move from ({old_position}) to ({new_position})", f"Indexes({obs_index}, {self.player_selector._index})")

        ball_pos = self.observation["ball_position"]
        if SoccerEnv.is_near(new_position, ball_pos, 15.0) \
            and ball_pos not in self.observation["left_team"] \
            and ball_pos not in self.observation["right_team"]\
            and self.player_selector._kickoff == False:

            # Autograb the ball if near enough and 
            # no player is in the same pos of the ball 
            self.observation["ball_position"] = new_position
            self.player_selector._kickoff = True
            self.player_selector._index = 10
            print("@@@@@@@@ Aconteceu kickoff @@@@@@@@")

        self.player_selector.next_()

        return self.observation, 0, False, False, {}
    

    def __initialize_players(self, num_agents = 22):

        def random_coordinates_generator(n: int = 22, gen_int: bool = True):
            if gen_int:
                x = np.random.randint(0, FIELD_WIDTH, size=n)
                y = np.random.randint(0, FIELD_HEIGHT, size=n)
            else:
                x = np.random.uniform(0, FIELD_WIDTH, size=n)
                y = np.random.uniform(0, FIELD_HEIGHT, size=n)

            return np.column_stack((x, y))
        
        # First 11 players will be left side players and last 11 will be right side
        self.all_start_positions = random_coordinates_generator() # posições de todos os jogadores
        self.player_names = ["player_" + str(r) for r in range(num_agents)]
        self.player_name_to_obs_index = dict( # mapping agent_name to index of observation channel
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
            field_image = self.field_drawer.draw_field(
                all_positions, 
                [self.observation["ball_position"]]
            )
            return field_image
            
        
        def rgb_array_render():
            left_team_positions = np.argwhere(self.observation[:, :, 0] > 0)
            left_team_positions = left_team_positions[:, ::-1]

            right_team_positions = np.argwhere(self.observation[:, :, 1] > 0)
            right_team_positions = right_team_positions[:, ::-1]

            all_positions = np.concatenate([left_team_positions, right_team_positions], axis=0)

            ball_position = np.argwhere(self.observation[:, :, 2] > 0)
            ball_position = ball_position[:, ::-1]

            field_image = self.field_drawer.draw_field(all_positions, ball_position)

            return field_image

        render_functions = {
            "human": human_render,
            "rgb_array": rgb_array_render 
        }
        self.render_function = render_functions[render_mode]


    def __initialize_observation_builder(self, observation_format: str) -> None:
        if observation_format == 'image':
            self.observation_builder = ImageObservationBuilder()
        elif observation_format == 'dict':
            self.observation_builder = DictObservationBuilder()
        else:
            raise ValueError("Invalid observation_type. Choose 'image' or 'dict'.")
       

    @staticmethod
    def is_near(pos_1: np.array, pos_2: np.array, threshold: float = 5.0):

        euclidian_distance = np.linalg.norm(pos_1 - pos_2)

        return euclidian_distance <= threshold
