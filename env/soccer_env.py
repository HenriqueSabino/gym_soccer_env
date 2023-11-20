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
        self.action_space = spaces.Discrete(13)
        self.observation_space = self._get_observation_space()


    def _get_observation_space(self):
        if self.observation_type == 'image':
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.field_height, self.field_width, 3),
                dtype=np.float32
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
            self.all_coordinates[:11],
            self.all_coordinates[11:22],
            self.all_coordinates[-1]
        )
        self.rewards = {agent: 0 for agent in self.player_names}
        self._cumulative_rewards = {agent: 0 for agent in self.player_names}
        self.terminations = {agent: False for agent in self.player_names}
        self.truncations = {agent: False for agent in self.player_names}
        self.infos = {agent: {} for agent in self.player_names}

        self.kickoff = False

        return self.observation, self.infos


    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        overview:
        [1] - Translate action
        [2] - Get select player to make action
        [3] - Apply action to selected player
        [4] - Change selected player
        [5] - Check kickoff logic
        """
        # If needed -> References to see step() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226


        # [1] - Translate action
        t_action = self.action_translator(action)
        
        # [2] - Get select player to make action
        player_name, direction, _, team  = self.player_selector.get_info()

        self.actions(action, team, t_action.direction, direction, player_name)
        
        # [4] - Change selected player
        self.player_selector.next_player()

        self.observation = self.observation_builder.build_observation(
            self.all_coordinates[:11],
            self.all_coordinates[11:22],
            self.all_coordinates[-1]
        )


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
        self.all_coordinates = random_coordinates_generator() # posições de todos os jogadores
        self.player_names = ["player_" + str(r) for r in range(num_agents)]
        
        # mapping agent_name to index of observation channel
        indexes = list(zip(
            list(range(11)) * 2, # Use in -> observation[team][index]
            list(range(22))      # Use in -> self.all_coordinates[index]
        ))
        self.player_name_to_obs_index = dict( 
            zip(self.player_names, indexes)
        )
        self.all_coordinates = np.vstack((self.all_coordinates, (np.array(POINTS["center"], dtype=np.float32))))
        self.player_directions = np.array([[1, 0]] * 11 + [[-1, 0]] * 11)

    def __initialize_action_translator(self, action_format):
        if action_format == 'discrete':
            self.action_translator = DiscreteActionTranslator.translate_action
        else:
            raise Exception("Action format is not recognized")
        

    def __initialize_render_function(self, render_mode: str = "humam") -> None:

        def human_render():
            left_team_positions = self.all_coordinates[:11]
            right_team_positions = self.all_coordinates[11:22]
            ball_position = self.all_coordinates[-1:]
            field_image = self.field_drawer.draw_field(
                list(left_team_positions) + list(right_team_positions), 
                ball_position
            )
            return field_image

            
        
        def rgb_array_render():
            if self.observation_type == 'image':
                left_team_positions = self.all_coordinates[:11]
                right_team_positions = self.all_coordinates[11:22]
                ball_position = self.all_coordinates[-1:]
                field_image = self.field_drawer.draw_field(
                    list(left_team_positions) + list(right_team_positions), 
                    ball_position
                )
                return field_image
            elif self.observation_type=='dict':
                array_1 = self.observation["left_team"]
                array_2 = self.observation["right_team"]
                all_positions = np.concatenate([array_1, array_2], axis=0)
                field_image = self.field_drawer.draw_field(
                    all_positions, 
                    [self.observation["ball_position"]]
                )
                return field_image

        render_functions = {
            "human": human_render,
            "rgb_array": rgb_array_render 
        }
        self.render_function = render_functions[render_mode]


    def __initialize_observation_builder(self, observation_format: str) -> None:
        if observation_format == 'image':
            self.observation_builder = ImageObservationBuilder(self.field_drawer.scale,
            self.field_drawer.border_size)
        elif observation_format == 'dict':
            self.observation_builder = DictObservationBuilder()
        else:
            raise ValueError("Invalid observation_type. Choose 'image' or 'dict'.")
       

    @staticmethod
    def apply_action_to_right_player(action, old_position):
        # To be use later in a refactor (don't delete yet)
        new_position = old_position + action * PLAYER_VELOCITY * [-1, 1]
        return new_position

    @staticmethod
    def apply_action_to_left_player(action, old_position):
        # To be use later in a refactor (don't delete yet)
        new_position = old_position + action * PLAYER_VELOCITY
        return new_position


    @staticmethod
    def is_near(pos_1: np.array, pos_2: np.array, threshold: float = 5.0):

        euclidian_distance = np.linalg.norm(pos_1 - pos_2)

        return euclidian_distance <= threshold

    def actions(self, action: int, team: str, t_action_direction: np.array, direction: np.array, player_name: str) -> None:
        obs_team_index, all_coordinates_index  = self.player_name_to_obs_index[player_name]
        ball_pos = self.all_coordinates[-1]
        is_new_position = False
        if action >=0 and action <= 7:
            old_position = self.all_coordinates[all_coordinates_index].copy()
            new_position = old_position + np.array(t_action_direction) * PLAYER_VELOCITY * direction
            new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))
            self.all_coordinates[all_coordinates_index] = new_position
            self.player_directions[all_coordinates_index] = t_action_direction
            print(team, player_name, f"move from ({old_position}) to ({new_position})", f"Indexes({all_coordinates_index}, {self.player_selector._index})")
            is_new_position = True
        elif action == 8:
            if (t_action_direction == np.array([2, 2])).all():
                if team == 'left_team':
                    if self.all_coordinates[-1] in self.all_coordinates[11:22]:
                        for i, coordendas in enumerate(self.all_coordinates[11:22]):
                            is_near = SoccerEnv.is_near(self.all_coordinates[all_coordinates_index],coordendas , 15.0)
                            if is_near and (self.all_coordinates[-1] == coordendas).all():
                                self.all_coordinates[-1] = self.all_coordinates[all_coordinates_index]
                                print(player_name,f"Roubou a bola de {self.player_names[i+11]}")
                else:
                    if self.all_coordinates[-1] in self.all_coordinates[:11]:
                        for i, coordendas in enumerate(self.all_coordinates[:11]):
                            is_near = SoccerEnv.is_near(self.all_coordinates[all_coordinates_index],coordendas , 15.0)
                            if is_near and (self.all_coordinates[-1] == coordendas).all():
                                self.all_coordinates[-1] = self.all_coordinates[all_coordinates_index]
                                print(player_name,f"Roubou a bola de {self.player_names[i]}")
        elif action == 9 and (self.all_coordinates[all_coordinates_index] == self.all_coordinates[-1]).all():
            old_position = self.all_coordinates[-1].copy()
            new_position = old_position + np.array(t_action_direction) * 1.5 * self.player_directions[all_coordinates_index]
            new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))
            self.all_coordinates[-1] = new_position
        elif action == 10 and (self.all_coordinates[all_coordinates_index] == self.all_coordinates[-1]).all():
            old_position = self.all_coordinates[-1].copy()
            new_position = old_position + np.array(t_action_direction) * 2.5 * self.player_directions[all_coordinates_index]
            new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))
            self.all_coordinates[-1] = new_position
        elif action == 11 and (self.all_coordinates[all_coordinates_index] == self.all_coordinates[-1]).all():
            old_position = self.all_coordinates[-1].copy()
            new_position = old_position + np.array(t_action_direction) * 3.5 * self.player_directions[all_coordinates_index]
            new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))
            self.all_coordinates[-1] = new_position
        
        

        # [5] - Check kickoff logic
        if is_new_position:
            if SoccerEnv.is_near(new_position, ball_pos, 15.0) \
                and ball_pos not in self.all_coordinates[:11] \
                and ball_pos not in self.all_coordinates[11:22]\
                and self.kickoff == False:
                    
                    # Autograb the ball if near enough and 
                    # no player is in the same pos of the ball 
                    self.all_coordinates[-1] = new_position
                    self.kickoff = True
                    self.player_selector.change_selector_logic()
                    print("@@@@@@@@ Aconteceu kickoff @@@@@@@@")