from typing import Union
from typing import Any, SupportsFloat
from pettingzoo import AECEnv
from gymnasium.core import RenderFrame
from gymnasium import spaces
from env.discrete_action_translator import DiscreteActionTranslator
from env.player_selection import PlayerSelector
from env.field_drawer import FieldDrawer
from env.image_observation_builder import ImageObservationBuilder
from env.dict_observation_builder import DictObservationBuilder
import numpy as np

from env.constants import FIELD_WIDTH, FIELD_HEIGHT, PLAYER_VELOCITY, POINTS


class SoccerEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4
    }

    def __init__(self, 
                 render_mode="rgb_array", 
                 action_format='discrete', 
                 observation_format='image', 
                 render_scale=8,
                 sparse_net_score_reward=False,
                 ball_posession_reward=False
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

        self.left_team_score = 0
        self.right_team_score = 0
        self.sparse_net_score_reward = sparse_net_score_reward
        self.ball_posession_reward = ball_posession_reward
        self.ball_posession = None
        self.last_ball_posession = None

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
        """
        overview:
        [1] - Initialize PlayerSelector
        [2] - Build observation data structure
        [3] - Define for all agents rewards, cumulative rewards, etc.
        [4] - Define global state variables
        """
        # If needed -> References to see reset() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226

        # [1] - Initialize PlayerSelector
        self.player_selector = PlayerSelector(self.player_names)

        player_name, _, _, _ = self.player_selector.get_info()
        _, all_coordinates_index = self.player_name_to_obs_index[player_name]
        
        # [2] - Build observation data structure
        self.observation = self.observation_builder.build_observation(
            self.all_coordinates[:11],
            self.all_coordinates[11:22],
            self.all_coordinates[-1],
            all_coordinates_index >= 11
        )
        
        # [3] - Define for all agents rewards, cumulative rewards, etc.
        self.rewards = {agent: 0 for agent in self.player_names}
        self._cumulative_rewards = {agent: 0 for agent in self.player_names}
        self.terminations = {agent: False for agent in self.player_names}
        self.truncations = {agent: False for agent in self.player_names}
        self.infos = {agent: {} for agent in self.player_names}

        # [4] - Define global state variables
        self.kickoff = False

        self.left_team_score = 0
        self.right_team_score = 0
        self.ball_posession = None
        self.last_ball_posession = None

        return self.observation, self.infos


    def step(self, action: tuple[int,int]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
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
        _, all_coordinates_index = self.player_name_to_obs_index[player_name]

        penalty = self.actions(action, team, t_action.direction, direction, player_name)
        
        # [4] - Change selected player
        self.player_selector.next_player()

        ball_pos = self.all_coordinates[-1]
        goal_size = 7.32

        left_team_goal = None
        # right team goal
        if (ball_pos[0] == 0
            and ball_pos[1] > self.field_height / 2 + goal_size / 2
            and ball_pos[1] > self.field_height / 2 - goal_size / 2):
            self.right_team_score += 1
            left_team_goal = False
            # TODO: kickoff
        elif (ball_pos[0] == self.field_width
            and ball_pos[1] > self.field_height / 2 + goal_size / 2
            and ball_pos[1] > self.field_height / 2 - goal_size / 2):
            self.left_team_score += 1
            left_team_goal = True
            # TODO: kickoff

        reward = penalty
    
        if self.ball_posession_reward:
            team_range = range(11) if team == "left_team" else range(11, 22)
            for player_pos in self.all_coordinates[team_range]:
                if player_pos == ball_pos:
                    reward += 0.1
                    break

        if not self.sparse_net_score_reward:
            if team == "left_team":
                reward += self.left_team_score - self.right_team_score
            else:
                reward += self.right_team_score - self.left_team_score

        elif self.sparse_net_score_reward:
            # XOR check
            if not(team == "left_team" ^ left_team_goal):
                reward += 1
            else:
                reward -= 1

        if self.last_ball_posession is not None and self.last_ball_posession != self.ball_posession:
            if team == self.ball_posession:
                reward += 0.01
            else:
                reward -= 0.01

        self.observation = self.observation_builder.build_observation(
            self.all_coordinates[:11],
            self.all_coordinates[11:22],
            self.all_coordinates[-1],
            all_coordinates_index >= 11
        )

        return self.observation, reward, False, False, {}
    

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
        
        # mapping agent_name to all indexes used in the code
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

    # implementado passe inteligente, roubar a bola. Falta implementar os chutes inteligentes e proteger a bola.
    # na tupla de ação, o primeiro elemento é a direção e o segundo é a ação, nas ações 0 é andar, 1 é roubar bola, 2 é passe, 3 é chute e 4 é proteger a bola
    # Falta conferir os valor fixos que estou utilizando, para ver de qual distancia pode roubar a bola, 
    # a distancia do passe e também a distancia que o jogador tem que estar da bola para ele receber o passe, 
    # também falta conferir a distancia do chute
    # retorna uma penalidade por executar ações indevidas, ou zero
    def actions(self, action: tuple[int,int], team: str, t_action_direction: np.array, direction: np.array, player_name: str) -> float:
        _, all_coordinates_index  = self.player_name_to_obs_index[player_name]

        player_pos = self.all_coordinates[all_coordinates_index]
        ball_pos = self.all_coordinates[-1]

        new_player_pos = None
        is_near = SoccerEnv.is_near(player_pos, ball_pos, 15.0)

        penalty = 0
        # player para enquanto executa ações
        if action[1] == 0:
            new_player_pos = self.__move_player(all_coordinates_index, t_action_direction, direction, team, player_name)
            player_pos = new_player_pos
        elif action[1] == 1:
            penalty = self.__steal_ball_action(all_coordinates_index, team, player_name)
        # Ação de passe inteligente implementado, conferindo o jogador mais proximo da localização em que a bola pararia apos o passe.
        elif action[1] == 2 and is_near:
            self.__pass_ball(t_action_direction, direction, team)
        elif action[1] == 3 and is_near:
            self.__kick_ball(all_coordinates_index)
        elif action[1] == 4:
            self.defend_ball()

        # [5] - Check kickoff logic
        if new_player_pos is not None and SoccerEnv.is_near(new_player_pos, ball_pos, 15.0) \
            and ball_pos not in self.all_coordinates[:11] \
            and ball_pos not in self.all_coordinates[11:22]\
            and self.kickoff == False:
                
                # Autograb the ball if near enough and 
                # no player is in the same pos of the ball 
                self.all_coordinates[-1] = new_player_pos
                self.ball_posession = team
                self.kickoff = True
                
                self.player_selector.change_selector_logic()
                print("@@@@@@@@ Aconteceu kickoff @@@@@@@@")
        
        if not is_near and action[1] >= 2:
            penalty = -0.01

        return penalty

    def __steal_ball_action(self, all_coordinates_index, team, player_name):
        if np.random.rand() < 0.5:
            steal_ball = True
            print("Tentou roubar bola")
        else:
            steal_ball = False
            print("Falhou ao tentar roubar bola") 

        if steal_ball:
            if team == 'left_team':
                if self.all_coordinates[-1] in self.all_coordinates[11:22]:
                    for i, coordendas in enumerate(self.all_coordinates[11:22]):
                        is_near = SoccerEnv.is_near(self.all_coordinates[all_coordinates_index],coordendas , 15.0)
                        if is_near and (self.all_coordinates[-1] == coordendas).all():
                            self.all_coordinates[-1] = self.all_coordinates[all_coordinates_index]
                            print(player_name,f"Roubou a bola de {self.player_names[i+11]}")
                            self.last_ball_posession = self.ball_posession
                            self.ball_posession = team
                        else:
                            return -0.01
                else:
                    return -0.01
            else:
                if self.all_coordinates[-1] in self.all_coordinates[:11]:
                    for i, coordendas in enumerate(self.all_coordinates[:11]):
                        is_near = SoccerEnv.is_near(self.all_coordinates[all_coordinates_index],coordendas , 15.0)
                        if is_near and (self.all_coordinates[-1] == coordendas).all():
                            self.all_coordinates[-1] = self.all_coordinates[all_coordinates_index]
                            print(player_name,f"Roubou a bola de {self.player_names[i]}")
                            self.last_ball_posession = self.ball_posession
                            self.ball_posession = team
                        else:
                            return -0.01
                else:
                    return -0.01
        return 0

    def __move_player(self, all_coordinates_index, t_action_direction, direction, team, player_name):
        old_position = self.all_coordinates[all_coordinates_index].copy()
        new_position = old_position + np.array(t_action_direction) * PLAYER_VELOCITY * direction
        new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))

        if (old_position == self.all_coordinates[-1]).all():
            self.all_coordinates[-1] = new_position

        self.all_coordinates[all_coordinates_index] = new_position
        self.player_directions[all_coordinates_index] = t_action_direction

        print(team, player_name, f"move from ({old_position}) to ({new_position})", f"Indexes({all_coordinates_index}, {self.player_selector._index})")

        return new_position

    def __pass_ball(self, t_action_direction, direction, team):
        old_position = self.all_coordinates[-1].copy()
        new_position = old_position + np.array(t_action_direction)* 2.5 * direction
        new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))
        if team == 'left_team':
            min_distance = float('inf')  
            nearest_player_index = -1
            for i, coordenadas in enumerate(self.all_coordinates[:11]):
                is_near = SoccerEnv.is_near(new_position,coordenadas , 15.0)
                if is_near:
                    distance = np.linalg.norm(new_position - coordenadas)
    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_player_index = i

            if nearest_player_index != -1:
                self.all_coordinates[-1] = self.all_coordinates[nearest_player_index]
            else:
                self.all_coordinates[-1] = new_position
        else:
            if self.all_coordinates[-1] in self.all_coordinates[11:22]:
                for i, coordendas in enumerate(self.all_coordinates[11:22]):
                    is_near = SoccerEnv.is_near(new_position,coordendas , 15.0)
                if is_near:
                    distance = np.linalg.norm(new_position - coordenadas)
    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_player_index = i

            if nearest_player_index != -1:
                self.all_coordinates[-1] = self.all_coordinates[nearest_player_index]
            else:
                self.all_coordinates[-1] = new_position

    def __kick_ball(self, all_coordinates_index):
        old_position = self.all_coordinates[-1].copy()
        new_position = old_position + 1.5 * self.player_directions[all_coordinates_index]
        new_position = np.clip(new_position, (0, 0), (self.field_height, self.field_width))
        self.all_coordinates[-1] = new_position

    def defend_ball(self):
        pass