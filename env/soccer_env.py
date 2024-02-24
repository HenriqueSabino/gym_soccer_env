from typing import Union
from typing import Any, SupportsFloat
from pettingzoo import AECEnv
from gymnasium.core import RenderFrame
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from env.action_translator import ActionTranslator
from env.discrete_action_translator import DiscreteActionTranslator
from env.player_selection import PlayerSelector
from env.field_drawer import FieldDrawer
from env.image_observation_builder import ImageObservationBuilder
from env.dict_observation_builder import DictObservationBuilder
import numpy as np
import itertools

from env.constants import \
    FIELD_WIDTH, FIELD_HEIGHT, POINTS, \
    PLAYER_VELOCITY, BALL_VELOCITY, \
    TEAM_LEFT_NAME, TEAM_RIGHT_NAME, \
    MID_FIELD_X, MID_FIELD_Y, \
    GOAL_SIZE, TOP_GOAL_Y, BOTTOM_GOAL_Y, CENTER_GOAL_Y, \
    OUTER_GOAL_HEIGHT, OUTER_GOAL_WIDTH, INNER_GOAL_HEIGHT, INNER_GOAL_WIDTH, \
    YELLOW, BRIGHT_PURPLE


class SoccerEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4, # Used to define pygame clock tick rate
        "name": "Soccer-v0"
    }

    def __init__(self, 
                 render_mode="rgb_array", 
                 render_scale=8,
                 action_format='discrete', 
                 observation_format='image', 
                 num_agents=22,
                 target_score=2, # First team to reach target_score ends the episode
                 sparse_net_score_reward=False,
                 ball_posession_reward=False,
                 color_option=0, # Availiable options: 0, 1, 2
                 left_start=True, # if true, left team start
                 control_goalkeeper = False,
                 first_kickoff_player_index = 0
                ) -> None:
        super().__init__()

        # If needed -> References to see __init__():
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226

        assert num_agents % 2 == 0, 'number of agents must be even'
        assert num_agents >= 2, 'number of agents must be greater than 2'
        
        self.render_mode = render_mode
        self.field_drawer = FieldDrawer(render_scale, border_size=2)

        self.number_agents = num_agents
        self.half_number_agents = num_agents // 2
        self.target_score = target_score
        self.__initialize_players(num_agents, left_start)
        self.__initialize_action_translator(action_format)
        self.__initialize_render_function(render_mode, color_option)
        self.__initialize_observation_builder(observation_format)
        self.observation_type = observation_format
        self.color_option = color_option
        self.left_start = left_start
        self.first_kickoff_player_index = first_kickoff_player_index
        self.control_goalkeeper = control_goalkeeper

        self.player_selector = PlayerSelector(
            self.agents, 
            left_start, 
            first_kickoff_player_index, 
            control_goalkeeper
        )

        # TODO: ver se observation_spaces action_spaces devem ficar com dict ou 
        # se basta a box sozinha
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    self._get_observation_spaces() 
                    for _ in enumerate(self.agents)
                ]
            )
        )
        self.action_spaces = dict(
            zip(
                self.agents,
                [
                    self.action_translator.action_space()
                    for _ in enumerate(self.agents)
                ]
            )
        )

        self.observation_space = self.observation_spaces[self.agents[0]]
        self.action_space =  self.action_translator.action_space()

        self.left_team_score = 0
        self.right_team_score = 0
        self.sparse_net_score_reward = sparse_net_score_reward
        self.ball_posession_reward = ball_posession_reward
        self.ball_posession = None
        self.last_ball_posession = None
        self.pygame_window = None
        self.pygame_clock = None

    def _get_observation_spaces(self):
        if self.observation_type == 'image':
            shape = (FIELD_WIDTH, FIELD_HEIGHT, 3)
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        elif self.observation_type == 'dict':
            return spaces.Dict({
                TEAM_LEFT_NAME: spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
                    dtype=np.float32,
                ),
                TEAM_RIGHT_NAME: spaces.Box(
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
        self.player_selector = PlayerSelector(
            self.agents, 
            self.left_start, 
            self.first_kickoff_player_index, 
            self.control_goalkeeper
        )

        player_name, _, _, _, _ = self.player_selector.get_info()
        _, all_coordinates_index = self.player_name_to_obs_index[player_name]
        
        # [2] - Build observation data structure
        self.observation = self.observation_builder.build_observation(
            # relative player index in team
            all_coordinates_index % (self.half_number_agents),
            self.all_coordinates[0:(self.half_number_agents)].copy(),
            self.all_coordinates[(self.half_number_agents):self.number_agents].copy(),
            self.all_coordinates[-1].copy(),
            all_coordinates_index >= (self.half_number_agents)
        )
        
        # [3] - Define for all agents rewards, cumulative rewards, etc.
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # [4] - Define global state variables
        self.is_before_kickoff = True

        self.left_team_score = 0
        self.right_team_score = 0
        self.ball_posession = None
        self.last_ball_posession = None

        # Must return:
        # [0] observation(ObsType): An element of the environment's observation_space as the next observation due to the agent actions.
        # [1] info(dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        return self.observation, self.infos
        


    def step(self, action: tuple[int,int]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        overview:
        [1] - Translate action
        [2] - Get select player to make action
        [3] - Removes, if defending, the current player of the defending list
        [4] - Apply action to selected player and Check kickoff logic
        [5] - Check goal logic
        [6] - Change selected player
        [7] - Calculate reward
        [8] - Update observation
        [9] - Render
        """
        # If needed -> References to see step() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226

        # [1] - Translate action
        t_action = self.action_translator.translate_action(action)
        
        # [2] - Get select player to make action
        player_name, foward_direction, _, team, other_team  = self.player_selector.get_info()
        _, all_coordinates_index = self.player_name_to_obs_index[player_name]

        # [3] - Removes, if defending, the current player of the defending list
        self.clean_defended_positions(player_name, team)

        # [4] - Apply action and Check kickoff logic
        penalty = self.actions(action, team, other_team, t_action.direction, foward_direction, player_name)
        self.goalkeeper(team)

        # [5] - Check goal logic
        # Remember: y axis grows down ([0,1] points down), hence "> top" & "< botton"
        ball_pos = self.all_coordinates[-1]

        left_team_goal = None
        # right team goal
        if (ball_pos[0] == 0
            and ball_pos[1] > TOP_GOAL_Y
            and ball_pos[1] < BOTTOM_GOAL_Y):
            self.right_team_score += 1
            left_team_goal = False
            self.is_before_kickoff = True
            self.player_selector.kickoff_rotation(TEAM_LEFT_NAME)
            # Reset position
            self.all_coordinates = self.start_positions_left_kickoff
            self.player_directions = self.start_directions
        elif (ball_pos[0] == FIELD_WIDTH
            and ball_pos[1] > TOP_GOAL_Y
            and ball_pos[1] < BOTTOM_GOAL_Y):
            self.left_team_score += 1
            left_team_goal = True
            self.is_before_kickoff = True
            self.player_selector.kickoff_rotation(TEAM_RIGHT_NAME)

            # Reset position
            self.all_coordinates = self.start_positions_right_kickoff
            self.player_directions = self.start_directions

        # [6] - Change selected player
        self.player_selector.next_player()

        # [7] - Calculate reward
        reward = penalty
    
        if self.ball_posession_reward:
            team_range = range(0,(self.half_number_agents)) if team == TEAM_LEFT_NAME else range((self.half_number_agents), self.number_agents)
            if SoccerEnv.is_in_any_array(ball_pos, self.all_coordinates[team_range]):
                reward += 0.1

        if not self.sparse_net_score_reward:
            if team == TEAM_LEFT_NAME:
                reward += self.left_team_score - self.right_team_score
            else:
                reward += self.right_team_score - self.left_team_score

        elif self.sparse_net_score_reward:
            # XOR check
            if not(team == TEAM_LEFT_NAME ^ left_team_goal):
                reward += 1
            else:
                reward -= 1

        if self.last_ball_posession is not None and self.last_ball_posession != self.ball_posession:
            if team == self.ball_posession:
                reward += 0.01
            else:
                reward -= 0.01

        # [8] - Update observation
        self.observation = self.observation_builder.build_observation(
            all_coordinates_index % (self.half_number_agents),
            self.all_coordinates[0:(self.half_number_agents)].copy(),
            self.all_coordinates[(self.half_number_agents):self.number_agents].copy(),
            self.all_coordinates[-1].copy(),
            all_coordinates_index >= (self.half_number_agents)
        )
        # if debug:
        #     print("=-= DEBUG =-=")
        #     print("=-= @@ observation @@ =-=")
        #     print(self.observation)
        #     print("=-= @@ all_coordinates @@ =-=")
        #     print("=-=  =-=")
        #     print(self.all_coordinates)
        #     print("=-= @@ player_directions @@ =-=")
        #     print(self.player_directions)
        #     print("=-= @@ action @@ =-=")
        #     print(action)
        #     print("=-= @@ t_action.direction @@ =-=")
        #     print(t_action.direction)
        #     print("=-= DEBUG =-=")

        # [9] - Render
        if self.render_mode == 'human':
            self.render()

        # [10] - Check end of episode
        terminated = (
            self.target_score == self.left_team_score or
            self.target_score == self.right_team_score
        )

        # Must return:
        # observation (ObsType): An element of the environment's observation_space as the next observation due to the agent actions.
        # reward (SupportsFloat): The reward as a result of taking the action.
        # terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
        # truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
        ##                # Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. 
        ##                # Can be used to end the episode prematurely before a terminal state is reached.
        ##                # If true, the user needs to call reset.
        # info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        return self.observation, reward, terminated, False, {}
    

    def __initialize_players(self, num_agents: int, left_start: bool):

        def random_coordinates_generator(n: int = 22) -> list[np.ndarray[np.float32]]:
            """
            n(int): Quantidade de coordenadas geradas
            """
            t = np.float32
            x = np.random.uniform(0, FIELD_WIDTH, size=n).astype(t)
            y = np.random.uniform(0, FIELD_HEIGHT, size=n).astype(t)

            coordinates_list = [
                np.array(coord, dtype=t) 
                for coord in zip(x, y)
            ]

            return coordinates_list
        
        def deterministic_coordinate_generator(
                n: int = 11, 
                left_start: bool = True,
                flip: bool = False,
                use_noise_in_goalkeeper_pos: bool = True
                )-> list[np.ndarray[np.float32]]:
            """
            n(int): Quantidade de coordenadas geradas
            left_start(bool): Caso True, faz o segundo jogador iniciar muito perto da bola.
            flip(bool): Caso True, Espelha as posições x para o outro lado do campo.
            """
            t = np.float32
            x_near = 1.0 if left_start else 0.8

            if use_noise_in_goalkeeper_pos:
                noise = np.random.uniform(FIELD_HEIGHT/2 + GOAL_SIZE/2, FIELD_HEIGHT/2 - GOAL_SIZE/2)
                goalkeeper_pos = np.array([0.1, noise], dtype=t)
            else:
                goalkeeper_pos = np.array([0.1, MID_FIELD_Y], dtype=t)
            
            # 11 positions to pick. if n > 11, then these will repeat
            # Players are positioned with some relative spacing
            # Note: Adjust the coordinates based on preferred formation nuances
            predefined_positions = [
                goalkeeper_pos,                                             # [0] Goalkeeper, slightly off the goal line at center
                np.array([MID_FIELD_X * 0.8, MID_FIELD_Y * 0.8 ], dtype=t), # [1] Forward up
                np.array([MID_FIELD_X*x_near, MID_FIELD_Y       ], dtype=t),# [2] Forward middle, slightly ahead for kickoff
                np.array([MID_FIELD_X * 0.8, MID_FIELD_Y * 1.2 ], dtype=t), # [3] Forward down
                np.array([MID_FIELD_X * 0.3, MID_FIELD_Y * 0.75], dtype=t), # [4] Defender up
                np.array([MID_FIELD_X * 0.3, MID_FIELD_Y * 1.25], dtype=t), # [5] Defender down
                np.array([MID_FIELD_X * 0.4, MID_FIELD_Y * 0.5 ], dtype=t), # [6] Defender up
                np.array([MID_FIELD_X * 0.4, MID_FIELD_Y * 1.5 ], dtype=t), # [7] Defender down
                np.array([MID_FIELD_X * 0.6, MID_FIELD_Y * 0.6 ], dtype=t), # [8] Midfielder up
                np.array([MID_FIELD_X * 0.6, MID_FIELD_Y       ], dtype=t), # [9] Midfielder middle
                np.array([MID_FIELD_X * 0.6, MID_FIELD_Y * 1.4 ], dtype=t), # [10] Midfielder down
            ]

            if flip:
                for i, (x, y) in enumerate(predefined_positions):
                    predefined_positions[i] = np.array((FIELD_WIDTH - x, y), dtype=t)

            # pick positions in a circular manner
            circular_positions = itertools.cycle(predefined_positions)
            players = []
            for _ in range(n):
                players.append(next(circular_positions))

            return players
        
        # First n players will be left side and last n will be right side
        left_team_coordinates = deterministic_coordinate_generator(num_agents // 2, left_start)
        right_team_coordinates = deterministic_coordinate_generator(num_agents // 2, not left_start, flip=True)
        self.all_coordinates = left_team_coordinates + right_team_coordinates
        self.agents = ["player_" + str(r) for r in range(self.number_agents)]
        
        # mapping agent_name to all indexes used in the code
        indexes = list(zip(
            list(range(self.half_number_agents)) * 2, # Use in -> observation[team][index]
            list(range(self.number_agents))           # Use in -> self.all_coordinates[index]
        ))
        self.player_name_to_obs_index = dict( 
            zip(self.agents, indexes)
        )

        # Add ball position
        self.all_coordinates.append(np.array(POINTS["center"], dtype=np.float32))

        # Initialize player directions and defended positions.
        self.player_directions = np.array([[1, 0]] * (self.half_number_agents) + [[-1, 0]] * (self.half_number_agents))
        self.defend_positions = {
            TEAM_LEFT_NAME: {
                "positions": [],
                "player_names": [],
            },
            TEAM_RIGHT_NAME: {
                "positions": [],
                "player_names": [],
            }
        }
        
        # Guarda a posição e direção inicial para resetar depois do gol
        self.start_positions_left_kickoff = self.all_coordinates.copy()
        self.start_positions_right_kickoff = self.start_positions_left_kickoff.copy()
        self.start_directions = self.player_directions

        # print('ALL COORDINATES:', self.all_coordinates)


    def __initialize_action_translator(self, action_format):
        if action_format == 'discrete':
            self.action_translator = DiscreteActionTranslator()
        else:
            raise Exception("Action format is not recognized")
        

    def __initialize_render_function(self, render_mode: str = "human", color_option: int = 0) -> None:
        
        # Site to visualize color schemas -> https://www.realtimecolors.com/?colors=050316-0f5415-24972e-c528f0-f9e636&fonts=Poppins-Poppins
        color_schemas = [
            {
                'field_bg_color': "green",
                'player_left_color': "red",
                'player_right_color': "Blue",
                'ball_color': "black"
            },
            {
                'field_bg_color': "green",
                'player_left_color': "purple",
                'player_right_color': "Blue",
                'ball_color': "black"
            },
            {
                'field_bg_color': "green",
                'player_left_color': BRIGHT_PURPLE,
                'player_right_color': YELLOW, 
                'ball_color': "black"
            }
        ]
        colors = color_schemas[color_option]

        def human_render():
            left_team_positions = self.all_coordinates[:(self.half_number_agents)]
            right_team_positions = self.all_coordinates[(self.half_number_agents):self.number_agents]
            ball_position = self.all_coordinates[-1:]
            field_image = self.field_drawer.draw_field(
                list(left_team_positions) + list(right_team_positions), 
                ball_position,
                **colors
            )

            field_image = np.array(field_image).transpose((1, 0, 2))

            """Fetch the last frame from the base environment and render it to the screen."""
            try:
                import pygame
            except ImportError as e:
                raise DependencyNotInstalled(
                    "pygame is not installed, run `pip install gymnasium[box2d]`"
                ) from e
            
            if self.pygame_window is None:
                pygame.init()
                pygame.display.init()
                self.pygame_window = pygame.display.set_mode(field_image.shape[:2])

            if self.pygame_clock is None:
                self.pygame_clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(field_image)
            self.pygame_window.blit(surf, (0, 0))
            pygame.event.pump()
            self.pygame_clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        
        def rgb_array_render():
            if self.observation_type == 'image':
                left_team_positions = self.all_coordinates[:(self.half_number_agents)]
                right_team_positions = self.all_coordinates[(self.half_number_agents):self.number_agents]
                ball_position = self.all_coordinates[-1:]
                field_image = self.field_drawer.draw_field(
                    list(left_team_positions) + list(right_team_positions), 
                    ball_position,
                    **colors
                )
                return field_image
            elif self.observation_type=='dict':
                array_1 = self.observation[TEAM_LEFT_NAME]
                array_2 = self.observation[TEAM_RIGHT_NAME]
                all_positions = np.concatenate([array_1, array_2], axis=0)
                field_image = self.field_drawer.draw_field(
                    all_positions, 
                    [self.observation["ball_position"]],
                    **colors
                )
                return field_image

        render_functions = {
            "human": human_render,
            "rgb_array": rgb_array_render 
        }

        self.render_function = render_functions[render_mode]


    def __initialize_observation_builder(self, observation_format: str) -> None:
        if observation_format == 'image':
            self.observation_builder = ImageObservationBuilder(
                self.field_drawer.scale,
                self.field_drawer.border_size
            )
        elif observation_format == 'dict':
            self.observation_builder = DictObservationBuilder(
                self.field_drawer.scale,
                self.field_drawer.border_size
            )
        else:
            raise ValueError("Invalid observation_type. Choose 'image' or 'dict'.")


    @staticmethod
    def is_in_any_array(some_position: np.ndarray, 
                        position_matrix: np.ndarray
                        ) -> bool:
        """
        Verifica se some_position é igual a pelo menos um dos arrays na pos_matrix.

        - some_position(np.ndarray): O array que você deseja verificar.
        - position_matrix(np.ndarray): Uma matriz contendo arrays de 2 elementos para comparação.
        
        Returns True if some_position is equal to at least one array in position_matrix, False otherwise.
        """
        return np.any(np.all(some_position == position_matrix, axis=1))


    @staticmethod
    def is_near(pos_1: np.array, 
                pos_2: np.array, 
                threshold: float = 5.0
                ) -> bool:

        euclidian_distance = np.linalg.norm(pos_1 - pos_2)

        return euclidian_distance <= threshold


    @staticmethod
    def angle_difference(angle1, angle2):
        return np.abs(np.angle(np.exp(1j * (angle1 - angle2))))
    

    @staticmethod
    def angle_between_vectors(u: np.ndarray[np.float32], 
                              v: np.ndarray[np.float32]
                              ) -> np.float32:
        dot_product = np.dot(u, v)
        magnitude_u = np.linalg.norm(u)
        magnitude_v = np.linalg.norm(v)

        cos_theta = dot_product / (magnitude_u * magnitude_v)
        theta = np.arccos(cos_theta)
        # theta = np.degrees(theta)

        return theta
    

    @staticmethod
    def rotate_vector(vector: np.ndarray[np.float32], 
                      angle_degrees: int
                      ) -> np.ndarray[np.float32]:

        # Convertendo o ângulo para radianos
        angle_radians: np.float64 = np.radians(angle_degrees)

        # Construindo a matriz de rotação
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                    [np.sin(angle_radians), np.cos(angle_radians)]])

        # Rotacionando o vetor em angle_degrees
        rotated_vector: np.ndarray = np.dot(rotation_matrix, vector)

        return rotated_vector
    

    @staticmethod
    def line_from_vector(vector: np.ndarray[np.float32]) -> tuple[np.float32, np.float32]:
        """
        Find the line equation "f(x) = mx + b" from a vector and \n
        returns the m and b values. \n
        A vector or 2 points is enough to define a line.
        """

        # Point 1 (x1, y1) and 2 (x2, y2) from input vector
        x1: np.float32 = 0
        y1: np.float32 = vector[0]
        x2: np.float32 = 1
        y2: np.float32 = vector[1]

        # Calculate slope (m)
        m = (y2 - y1) / (x2 - x1)
        m = -m

        # Calculate height when x = 0 (b)
        b = y1 + m * x1

        return m, b
    
    @staticmethod
    def normalize_direction(direction: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        if np.all(direction == 0):
            return direction
        
        return direction / np.linalg.norm(direction)
    
    
    # implementado passe inteligente, roubar a bola. Falta implementar os chutes inteligentes e proteger a bola.
    # na tupla de ação, o primeiro elemento é a direção e o segundo é a ação, nas ações 0 é andar, 1 é roubar bola, 2 é passe, 3 é chute e 4 é proteger a bola
    # Falta conferir os valor fixos que estou utilizando, para ver de qual distancia pode roubar a bola, 
    # a distancia do passe e também a distancia que o jogador tem que estar da bola para ele receber o passe, 
    # também falta conferir a distancia do chute
    # retorna uma penalidade por executar ações indevidas, ou zero
    def actions(self, action: tuple[int,int], team: str, other_team: str, t_action_direction: np.array, foward_direction: np.array, player_name: str) -> float:
        _, all_coordinates_index  = self.player_name_to_obs_index[player_name]

        player_pos = self.all_coordinates[all_coordinates_index]
        ball_pos = self.all_coordinates[-1]

        new_player_pos = None
        is_near = SoccerEnv.is_near(player_pos, ball_pos, 15.0)
        has_the_ball = (player_pos == ball_pos).all()

        penalty = 0
        # player para enquanto executa ações
        if action[1] == 0:
            new_player_pos = self.__move_player(all_coordinates_index, t_action_direction, foward_direction, team, player_name)
            # player_pos = new_player_pos
        elif action[1] == 1:
            penalty = self.__steal_ball_action(all_coordinates_index, team, other_team, player_name)
        # Ação de passe inteligente implementado, conferindo o jogador mais proximo da localização em que a bola pararia apos o passe.
        elif action[1] == 2 and has_the_ball:
            self.__pass_ball(t_action_direction, foward_direction, team)
        elif action[1] == 3 and has_the_ball:
            self.__kick_ball(all_coordinates_index, t_action_direction, action[0], foward_direction, team)
        elif action[1] == 4:
            self.__defend_position(self.all_coordinates[all_coordinates_index], team, player_name)

        # [4] - Check kickoff logic
        if (self.is_before_kickoff
            and new_player_pos is not None 
            and SoccerEnv.is_near(player_pos, ball_pos, 1.2)):
                
                # Autograb the ball if near enough and 
                # no player is in the same pos of the ball 
                self.all_coordinates[-1] = new_player_pos
                self.ball_posession = team
                self.is_before_kickoff = False
                
                self.player_selector.playing_rotation()
                print("@@@@@@@@ Aconteceu kickoff @@@@@@@@")
        
        if not is_near and action[1] >= 2:
            penalty = -0.01

        return penalty


    def __steal_ball_action(self, all_coordinates_index: int, team: str, other_team: str, player_name: str):
        
        player_position = self.all_coordinates[all_coordinates_index]
        defended_positions = self.defend_positions[other_team]["positions"]

        defense_bonus = +0.1 if SoccerEnv.is_near(player_position, defended_positions, 3.0) else 0
        if np.random.rand() < 0.5 + defense_bonus:
            steal_ball = True
            print("Tentou roubar bola")
        else:
            steal_ball = False
            print("Falhou ao tentar roubar bola") 

        if steal_ball:
            if team == TEAM_LEFT_NAME:
                if SoccerEnv.is_in_any_array(self.all_coordinates[-1], self.all_coordinates[(self.half_number_agents):self.number_agents]):
                    for i, coordendas in enumerate(self.all_coordinates[(self.half_number_agents):self.number_agents]):
                        is_near = SoccerEnv.is_near(player_position, coordendas, 15.0)
                        if is_near and (self.all_coordinates[-1] == coordendas).all():
                            self.all_coordinates[-1] = player_position
                            print(player_name,f"Roubou a bola de {self.agents[i+(self.half_number_agents)]}")
                            self.last_ball_posession = self.ball_posession
                            self.ball_posession = team
                        else:
                            return -0.01
                else:
                    return -0.01
            else:
                if SoccerEnv.is_in_any_array(self.all_coordinates[-1], self.all_coordinates[0:(self.half_number_agents)]):
                    for i, coordendas in enumerate(self.all_coordinates[0:(self.half_number_agents)]):
                        is_near = SoccerEnv.is_near(player_position, coordendas, 15.0)
                        if is_near and (self.all_coordinates[-1] == coordendas).all():
                            self.all_coordinates[-1] = player_position
                            print(player_name,f"Roubou a bola de {self.agents[i]}")
                            self.last_ball_posession = self.ball_posession
                            self.ball_posession = team
                        else:
                            return -0.01
                else:
                    return -0.01
        return 0


    def __move_player(self, 
                      all_coordinates_index: int, 
                      t_action_direction: np.ndarray, 
                      foward_direction: np.ndarray, 
                      team: str, 
                      player_name: str
                    ):
        # Calcula nova posição
        old_position = self.all_coordinates[all_coordinates_index].copy()
        new_position = old_position + t_action_direction * PLAYER_VELOCITY * foward_direction
        new_position = np.clip(new_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))

        # Faz a posição da bola (A.K.A self.all_coordinates[-1]) acompanhar o jogador
        if (old_position == self.all_coordinates[-1]).all():
            self.all_coordinates[-1] = new_position

        # Atualiza posição do jogador
        self.all_coordinates[all_coordinates_index] = new_position
        self.player_directions[all_coordinates_index] = t_action_direction

        print(team, player_name, f"move from ({old_position}) to ({new_position})", f"Indexes({all_coordinates_index}, {self.player_selector._index})")

        return new_position


    def __pass_ball(self, 
                    t_action_direction: np.ndarray, 
                    foward_direction: np.ndarray, 
                    team: str
                   ):
        old_position = self.all_coordinates[-1].copy()
        new_position = old_position + np.array(t_action_direction) * 10.5 * foward_direction
        print("forward_direction", foward_direction)
        print("t_action_direction", t_action_direction)
        new_position = np.clip(new_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))

        # Calcula o ângulo entre a direção do passe e a posição atual do jogador
        angle_center = np.arctan2(t_action_direction[1], t_action_direction[0]) + np.arctan2(foward_direction[1], foward_direction[0])
        print("Angle center", angle_center)

        if team == TEAM_LEFT_NAME:
            players_range = range(11)
        else:
            players_range = range(11, 22)

        min_angle_difference = float('inf')
        nearest_player_index = -1

        for i in players_range:
            coordenadas = self.all_coordinates[i]
            is_near = SoccerEnv.is_near(old_position, coordenadas, 15.0)
            #print("is_near", is_near)
            if is_near:
                vector_to_player = coordenadas - old_position
                player_distance = np.linalg.norm(vector_to_player)
                #print("vector_to_player", vector_to_player)
                #print("player_distance", player_distance)
                #print("coordenadas", coordenadas)

                if player_distance != 0:  # Evita divisão por zero
                    vector_to_player /= player_distance  # Normaliza o vetor para ter comprimento 1
                    dot_product = np.dot(t_action_direction, vector_to_player)
                    #print("dot_product", dot_product)
                    if dot_product >= 0:  # Verifica se o jogador está na direção do passe
                    # Calcula o ângulo entre o vetor posição do jogador e a direção do passe
                        angle_to_player = np.arctan2(vector_to_player[1], vector_to_player[0])
                        angle_difference = np.abs(self.angle_difference(angle_to_player, angle_center))
                        #print("angle_difference", angle_difference)
                        if angle_difference < min_angle_difference or (angle_difference == min_angle_difference and player_distance < min_distance_to_passer):
                            min_angle_difference = angle_difference
                            nearest_player_index = i
                            min_distance_to_passer = player_distance
    
        if nearest_player_index != -1:
            self.all_coordinates[-1] = self.all_coordinates[nearest_player_index]
        else:
            self.all_coordinates[-1] = new_position


    def __kick_ball(self, 
                    all_coordinates_index: int, 
                    t_action_direction: np.ndarray,
                    direction_index: int,
                    foward_direction: np.array,
                    team: str
                   ):
        
        player_position: np.ndarray[np.float32] = self.all_coordinates[all_coordinates_index]
        old_ball_position = self.all_coordinates[-1].copy()

        if direction_index in [1, 5, 7]:
            # Escolheu chutar para trás
            final_direction = t_action_direction * foward_direction
        elif direction_index == 8:
            # Sem direção escolhida, então chuta para direção atual do jogador
            final_direction = self.player_directions[all_coordinates_index] * foward_direction
            final_direction = SoccerEnv.normalize_direction(final_direction)
        else:

            # Team side adjustments
            if team == TEAM_LEFT_NAME:
                tiangle_x_lenght = FIELD_WIDTH - player_position[0]
                center_goal_x = FIELD_WIDTH
            else:
                tiangle_x_lenght = player_position[0]
                center_goal_x = 0
                
            # change y signal to make calculations considering y growing up instead of down
            t_action_direction[1] *= -1 

            # [1] - Rotacionar t_action_direction em + e - 45 graus
            rotation_angle = 45
            upper_vector = SoccerEnv.rotate_vector(t_action_direction, angle_degrees= rotation_angle)
            lower_vector = SoccerEnv.rotate_vector(t_action_direction, angle_degrees= -rotation_angle)

            # [2] - Encontrar b quando x = 0
            """ Explanation
            Consider RIGHT_TEAM player near goal with position (player_x, player_y)
            (0,0) * - - - - - - - - - - - - - - - -> (x axis)
                  |                      |  | y(x) = a * x + b defined by direction vector
                  |                      |  | small height y_0 = b = y(0)
                  @ - - - - - - - - - - -| y_0  - 
                  |\ ) alpha = 45º      L|      |
                  | \                    |      |
                  |  \                   |      |
                  |   \                  |      |
                  |    \                 |      |
                  |     \                |      |
                  |      \               |      |
                  |       \              |      |
                  |        \             |      | tiangle_x_lenght = player_x or mirrored(player_x) depending on team
                  |         \            |      | tan(alpha) = delta / tiangle_x_lenght
                  |          \           |      | delta_y = tiangle_x_lenght * tan(alpha)
                  |           \          |      |
                  |            \         |      |
                  |             \        |      |
                  |              \       |      |   foward direction = (-1, 1)
                  |               \      |      |   foward_direction can't be used because y != 0
                  |                \     |      |   horizontal_vector = (1,0)
                  |                 \    |      |   All possible alphas = [+-90, +-45, 0]
                  |                  \   |      |   alpha = angle_between(upper_direction, horizontal_vector)
                  |         45º=alpha(\  |      |   alpha = angle_between(lower_direction, horizontal_vector)
                  - - (player_x, player_y)-     -   
                  |
                  v (y axis)

            tan(90) = 1.995
            tan(45) = 1
            tan(0) = 0
            tan(135) = tan(-45) = -1
            tan(270) = tan(-90) = -1.995
            """
            # Original code version
            horizontal_vector = np.array((1,0), dtype=np.float32)
            # horizontal_vector = horizontal_vector * foward_direction
            
            upper_angle_radians = SoccerEnv.angle_between_vectors(upper_vector, horizontal_vector)
            if upper_vector[1] < 0:
                upper_angle_radians *= -1
            tan = np.tan(upper_angle_radians) 
            delta = tiangle_x_lenght * tan
            top_y_intersection = player_position[1] - delta
            
            lower_angle_radians = SoccerEnv.angle_between_vectors(lower_vector, horizontal_vector)
            if lower_vector[1] < 0:
                lower_angle_radians *= -1
            tan = np.tan(lower_angle_radians)
            delta = tiangle_x_lenght * tan
            bottom_y_intersection = player_position[1] - delta
            # bottom_y_intersection_2 = (player_position[1] - upper_y_intersection) + player_position[1]
            # assert bottom_y_intersection == bottom_y_intersection_2 # True!

            # TODO change old optimized version to work with more angles than +-45
            # Optimized code version (simplified because alpha is in [+90, +45, 0, -45, -90])
            # top_y_intersection = player_position[1] - player_position[0]
            # bottom_y_intersection = player_position[1] + player_position[0]

            can_kick_to_goal = \
                (upper_vector[0] > 0 and CENTER_GOAL_Y >= top_y_intersection) \
                or \
                (lower_vector[0] > 0 and CENTER_GOAL_Y <= bottom_y_intersection)
            
            # print(f"player_position: {player_position}")
            # print(f"horizontal_vector: {horizontal_vector}")
            # print(f"t_action_direction: {t_action_direction}")
            # print(f"upper_vector: {upper_vector}")
            # print(f"upper_angle_degree: {np.degrees(upper_angle_radians)}")
            # print(f"lower_vector: {lower_vector}")
            # print(f"lower_angle_degree: {np.degrees(lower_angle_radians)}")
            # print(f"top_y_intersection: {top_y_intersection}")
            # print(f"bottom_y_intersection: {bottom_y_intersection}")
            # print(f"CENTER_GOAL_Y: {CENTER_GOAL_Y}")
            # print(f"pode chutar ? {can_kick_to_goal}")

            if can_kick_to_goal:
                center_goal_point = np.array((center_goal_x, CENTER_GOAL_Y), dtype=np.float32)
                final_direction = center_goal_point - player_position
                final_direction = SoccerEnv.normalize_direction(final_direction)
            else:
                t_action_direction[1] *= -1 # Revert y signal to consider y growing down
                final_direction = t_action_direction  * foward_direction

        # Calculate and update ball position
        new_ball_position = old_ball_position + BALL_VELOCITY * final_direction
        new_ball_position = np.clip(new_ball_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))
        self.all_coordinates[-1] = new_ball_position



    def goalkeeper(self, team):
        """
        Makes the goalkeeper of the playing team walk randomly on the y-axis
        """
        if team == TEAM_LEFT_NAME:
            old_position = self.all_coordinates[0].copy()

            # Limite superior e inferior da linha do gol
            upper_limit = FIELD_HEIGHT // 2 + OUTER_GOAL_HEIGHT // 2
            lower_limit = FIELD_HEIGHT // 2 - OUTER_GOAL_HEIGHT // 2

            # Movimentação aleatória para cima ou para baixo
            move_direction = np.random.choice([-1, 1])

            # Distância aleatória para se movimentar (você pode ajustar a amplitude conforme necessário)
            move_distance = np.random.uniform(0, 1.5)

            # Calculando nova posição y
            new_position_y = old_position[1] + move_direction * move_distance
            new_position_y = np.clip(new_position_y, lower_limit, upper_limit)

            # Atualizando a posição apenas na coordenada y
            new_position = old_position.copy()
            new_position[1] = new_position_y

            # Atualizando self.all_coordinates para a posição do goleiro
            self.all_coordinates[0] = new_position
        elif team == TEAM_RIGHT_NAME:
            old_position = self.all_coordinates[(self.half_number_agents)].copy()

            # Limite superior e inferior da linha do gol
            upper_limit = FIELD_HEIGHT // 2 + OUTER_GOAL_HEIGHT // 2
            lower_limit = FIELD_HEIGHT // 2 - OUTER_GOAL_HEIGHT // 2
            # Movimentação aleatória para cima ou para baixo
            move_direction = np.random.choice([-1, 1])

            # Distância aleatória para se movimentar (você pode ajustar a amplitude conforme necessário)
            move_distance = np.random.uniform(0, 1.5)

            # Atualizando a posição apenas na coordenada y
            new_position_y = old_position[1] + move_direction * move_distance
            new_position_y = np.clip(new_position_y, lower_limit, upper_limit)

            # Atualizando a posição apenas na coordenada y
            new_position = old_position.copy()
            new_position[1] = new_position_y

            # Atualizando self.all_coordinates para a posição do goleiro
            self.all_coordinates[(self.half_number_agents)] = new_position


    def __defend_position(self, player_position: np.array, team: str, player_name: str):
        """
        Makes the player defend the position by adding its coordinates into a list.
        This list is used inside steal_ball action to reduce the odds of stealing.
        Each team has its list and all defended positions are cleaned after some turns.
        """
        self.defend_positions[team]["positions"].append(player_position)
        self.defend_positions[team]["player_names"].append(player_name)


    def clean_defended_positions(self, player_name: str, team: str):
        """
        Checks if the current player name is in a defended position and removes it.
        If a player takes the defend ball action, then it will be added back to the list.
        """
        defending_players_names = self.defend_positions[team]["player_names"]
        if player_name in defending_players_names:
            index = defending_players_names.index(player_name)
            self.defended_positions[team]["positions"].pop(index)
            self.defended_positions[team]["player_names"].pop(index)

