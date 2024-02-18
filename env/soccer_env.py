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

from env.constants import FIELD_WIDTH, FIELD_HEIGHT, PLAYER_VELOCITY, POINTS, TEAM_LEFT_NAME, TEAM_RIGHT_NAME, YELLOW, BRIGHT_PURPLE


class SoccerEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4
    }

    def __init__(self, 
                 render_mode="rgb_array", 
                 action_format='discrete', 
                 observation_format='image', 
                 num_agents=22,
                 render_scale=8,
                 sparse_net_score_reward=False,
                 ball_posession_reward=False,
                 color_option=0 # Availiable options: 0, 1, 2
                ) -> None:
        super().__init__()

        assert num_agents % 2 == 0, 'number of agents must be even'
        assert num_agents >= 2, 'number of agents must be greater than 2'
        
        self.render_mode = render_mode
        self.field_drawer = FieldDrawer(render_scale, border_size=2)

        self.number_agents = num_agents
        self.__initialize_players(num_agents=num_agents)
        self.__initialize_action_translator(action_format)
        self.__initialize_render_function(render_mode, color_option)
        self.__initialize_observation_builder(observation_format)
        self.observation_type = observation_format

        self.player_selector = PlayerSelector(self.player_names, kickoff_player_index=3)

        # TODO: verificar a definição de observaiton space e action space
        # # precisa conter o domínio (conjunto) de valores possíveis da observação
        # # Se uma observação não estivar dentro desse domínio, então vai sair um warning no terminal. 
        # # Não vai dar erro e tudo funciona normalmente.
        # # References to see __init__():
        # # https://pettingzoo.farama.org/content/environment_creation/
        # # https://www.gymlibrary.dev/content/environment_creation/
        # # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
        self.action_space = self.action_translator.action_space()
        self.observation_space = self._get_observation_space()

        self.left_team_score = 0
        self.right_team_score = 0
        self.sparse_net_score_reward = sparse_net_score_reward
        self.ball_posession_reward = ball_posession_reward
        self.ball_posession = None
        self.last_ball_posession = None
        self.pygame_window = None
        self.pygame_clock = None

    def _get_observation_space(self):
        if self.observation_type == 'image':
            return spaces.Box(
                low=0,
                high=255,
                shape=(FIELD_HEIGHT, FIELD_WIDTH, 3),
                dtype=np.float32
            )
        elif self.observation_type == 'dict':
            return spaces.Dict({
                TEAM_LEFT_NAME: spaces.Box(
                    low=np.array([0, 0], dtype=np.float64),
                    high=np.array([FIELD_HEIGHT, FIELD_WIDTH], dtype=np.float64),
                    dtype=np.float64,
                ),
                TEAM_RIGHT_NAME: spaces.Box(
                    low=np.array([0, 0], dtype=np.float64),
                    high=np.array([FIELD_HEIGHT, FIELD_WIDTH], dtype=np.float64),
                    dtype=np.float64,
                ),
                "ball_position": spaces.Box(
                    low=np.array([0, 0], dtype=np.float64),
                    high=np.array([FIELD_HEIGHT, FIELD_WIDTH], dtype=np.float64),
                    dtype=np.float64,
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
        self.player_selector = PlayerSelector(self.player_names, kickoff_player_index=3)

        player_name, _, _, _, _ = self.player_selector.get_info()
        _, all_coordinates_index = self.player_name_to_obs_index[player_name]
        
        # [2] - Build observation data structure
        self.observation = self.observation_builder.build_observation(
            # relative player index in team
            all_coordinates_index % (self.number_agents // 2),
            self.all_coordinates[0:(self.number_agents // 2)].copy(),
            self.all_coordinates[(self.number_agents // 2):self.number_agents].copy(),
            self.all_coordinates[-1].copy(),
            all_coordinates_index >= (self.num_agents // 2)
        )
        
        # [3] - Define for all agents rewards, cumulative rewards, etc.
        self.rewards = {agent: 0 for agent in self.player_names}
        self._cumulative_rewards = {agent: 0 for agent in self.player_names}
        self.terminations = {agent: False for agent in self.player_names}
        self.truncations = {agent: False for agent in self.player_names}
        self.infos = {agent: {} for agent in self.player_names}

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
        [8] - Update observatio
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
        ball_pos = self.all_coordinates[-1]
        goal_size = 7.32

        left_team_goal = None
        # right team goal
        if (ball_pos[0] == 0
            and ball_pos[1] < FIELD_HEIGHT / 2 + goal_size / 2
            and ball_pos[1] > FIELD_HEIGHT / 2 - goal_size / 2):
            self.right_team_score += 1
            left_team_goal = False
            self.is_before_kickoff = True
            self.player_selector.kickoff_rotation(TEAM_LEFT_NAME)
            # Reset position
            self.all_coordinates = self.start_positions_left_kickoff
            self.player_directions = self.start_directions
        elif (ball_pos[0] == FIELD_WIDTH
            and ball_pos[1] < FIELD_HEIGHT / 2 + goal_size / 2
            and ball_pos[1] > FIELD_HEIGHT / 2 - goal_size / 2):
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
            team_range = range(0,(self.number_agents // 2)) if team == TEAM_LEFT_NAME else range((self.number_agents // 2), self.number_agents)
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

        # [8] - Update observatio
        self.observation = self.observation_builder.build_observation(
            all_coordinates_index % (self.number_agents // 2),
            self.all_coordinates[0:(self.number_agents // 2)].copy(),
            self.all_coordinates[(self.number_agents // 2):self.number_agents].copy(),
            self.all_coordinates[-1].copy(),
            all_coordinates_index >= (self.num_agents // 2)
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

        # Must return:
        # observation (ObsType): An element of the environment's observation_space as the next observation due to the agent actions.
        # reward (SupportsFloat): The reward as a result of taking the action.
        # terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
        # truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
        ##                # Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. 
        ##                # Can be used to end the episode prematurely before a terminal state is reached.
        ##                # If true, the user needs to call reset.
        # info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        return self.observation, reward, False, False, {}
    

    def __initialize_players(self, num_agents = 22):

        def random_coordinates_generator(n: int = 22, gen_int: bool = True):
            """
            n(int): Quantidade de coordenadas geradas
            gen_int(bool): gera coordenadas do tipo dtype[int] ou dtype[float64]
            """
            if gen_int:
                x = np.random.randint(0, FIELD_WIDTH, size=n)
                y = np.random.randint(0, FIELD_HEIGHT, size=n)
            else:
                x = np.random.uniform(0, FIELD_WIDTH, size=n)
                y = np.random.uniform(0, FIELD_HEIGHT, size=n)

            return np.column_stack((x, y))
        
        def deterministic_coordinate_generator(n: int = 22, grid_size: int = 20):
            """
            n(int): Quantidade de coordenadas geradas
            grid_size(int): Quanto maior, mais juntos os pontos ficam
            """
            # Initialize an empty list for player positions

            midfield_x = FIELD_WIDTH // 2
            midfield_y = FIELD_HEIGHT // 2

            # Players' positioning
            # Team A (Player indexes 0-10)
            # Players are positioned with some relative spacing
            # Note: Adjust the coordinates based on preferred formation nuances
            players = [
                np.array([midfield_x * 0.1, midfield_y]),  # [0] Goalkeeper, slightly off the goal line at center
                np.array([midfield_x, midfield_y]),  # [1] Player exactly at midfield; this is the second player
            ]

            optional_players_pos_list = [
                np.array([midfield_x * 0.8, midfield_y * 0.8]),  # [10] Forward
                np.array([midfield_x * 0.4, midfield_y * 0.5]),  # [4] Defender
                np.array([midfield_x * 0.4, midfield_y * 1.5]),  # [5] Defender
                np.array([midfield_x * 0.3, midfield_y * 0.75]), # [2] Defender
                np.array([midfield_x * 0.3, midfield_y * 1.25]), # [3] Defender
                np.array([midfield_x * 0.6, midfield_y * 0.6]),  # [6] Midfielder
                np.array([midfield_x * 0.6, midfield_y]),        # [7] Midfielder, slightly ahead for kickoff
                np.array([midfield_x * 0.6, midfield_y * 1.4]),  # [8] Midfielder
                np.array([midfield_x * 0.8, midfield_y * 1.2]),  # [9] Midfielder
            ]

            for i in range(n // 2 - 2):
                players.append(optional_players_pos_list[i])
                
            
            players += [
                np.array([FIELD_WIDTH - players[0][0], players[0][1]]), 
                np.array([FIELD_WIDTH - midfield_x * 0.8, midfield_y]),  # [12] Player exactly at midfield; this is the second player
                ] + [
                np.array([FIELD_WIDTH - p[0], p[1]]) for p in players[2:(n//2)] # [13]-[21]
            ]

            return players
        
        # First 11 players will be left side and last 11 will be right side
        self.all_coordinates = deterministic_coordinate_generator(num_agents) # posições de todos os jogadores
        self.player_names = ["player_" + str(r) for r in range(self.number_agents)]
        
        # mapping agent_name to all indexes used in the code
        indexes = list(zip(
            list(range(self.number_agents // 2)) * 2, # Use in -> observation[team][index]
            list(range(self.number_agents))      # Use in -> self.all_coordinates[index]
        ))
        self.player_name_to_obs_index = dict( 
            zip(self.player_names, indexes)
        )
        self.outer_goal_height = 40 
        self.outer_goal_width = 16.5 

        self.inner_goal_height = 18.3 
        self.inner_goal_width = 5.5
        self.goal_size = 7.32

        self.all_coordinates[0] = np.array([
            0,
            np.random.uniform(FIELD_HEIGHT/2 + self.goal_size/2, FIELD_HEIGHT/2 - self.goal_size/2)
        ])

        self.all_coordinates[self.number_agents // 2] = np.array([
            FIELD_WIDTH,
            np.random.uniform(FIELD_HEIGHT/2 + self.goal_size/2, FIELD_HEIGHT/2 - self.goal_size/2)
        ])

        self.all_coordinates.append(np.array(POINTS["center"], dtype=np.float32))

        self.player_directions = np.array([[1, 0]] * (self.number_agents // 2) + [[-1, 0]] * (self.number_agents // 2))
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
        
        # Guarda a posição e direção inicial para usar depois do gol
        self.start_positions_left_kickoff = self.all_coordinates.copy()

        self.start_positions_right_kickoff = self.start_positions_left_kickoff.copy()

        self.start_positions_right_kickoff[(self.number_agents // 2 + 1)] = np.array([FIELD_WIDTH // 2, FIELD_HEIGHT // 2])
        self.start_positions_right_kickoff[1][0] *= 0.8

        # Essa linha faz o time azul começar com a bola
        self.all_coordinates = self.start_positions_right_kickoff
        
        print('ALL COORDINATES:',self.all_coordinates)

        self.start_directions = self.player_directions


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
            left_team_positions = self.all_coordinates[:(self.number_agents // 2)]
            right_team_positions = self.all_coordinates[(self.number_agents // 2):self.number_agents]
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
                left_team_positions = self.all_coordinates[:(self.number_agents // 2)]
                right_team_positions = self.all_coordinates[(self.number_agents // 2):self.number_agents]
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
            self.observation_builder = DictObservationBuilder()
        else:
            raise ValueError("Invalid observation_type. Choose 'image' or 'dict'.")


    @staticmethod
    def is_in_any_array(some_position: np.ndarray, position_matrix: np.ndarray) -> bool:
        """
        Verifica se some_position é igual a pelo menos um dos arrays na pos_matrix.

        - some_position(np.ndarray): O array que você deseja verificar.
        - position_matrix(np.ndarray): Uma matriz contendo arrays de 2 elementos para comparação.
        
        Returns True if some_position is equal to at least one array in position_matrix, False otherwise.
        """
        return np.any(np.all(some_position == position_matrix, axis=1))


    @staticmethod
    def is_near(pos_1: np.array, pos_2: np.array, threshold: float = 5.0) -> bool:

        euclidian_distance = np.linalg.norm(pos_1 - pos_2)

        return euclidian_distance <= threshold

    @staticmethod
    def angle_difference(angle1, angle2):
        return np.abs(np.angle(np.exp(1j * (angle1 - angle2))))
    
    
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
            self.__kick_ball(all_coordinates_index, foward_direction)
        elif action[1] == 4:
            self.defend_position(self.all_coordinates[all_coordinates_index], team, player_name)

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
                if SoccerEnv.is_in_any_array(self.all_coordinates[-1], self.all_coordinates[(self.number_agents // 2):self.number_agents]):
                    for i, coordendas in enumerate(self.all_coordinates[(self.number_agents // 2):self.number_agents]):
                        is_near = SoccerEnv.is_near(player_position, coordendas, 15.0)
                        if is_near and (self.all_coordinates[-1] == coordendas).all():
                            self.all_coordinates[-1] = player_position
                            print(player_name,f"Roubou a bola de {self.player_names[i+(self.number_agents // 2)]}")
                            self.last_ball_posession = self.ball_posession
                            self.ball_posession = team
                        else:
                            return -0.01
                else:
                    return -0.01
            else:
                if SoccerEnv.is_in_any_array(self.all_coordinates[-1], self.all_coordinates[0:(self.number_agents // 2)]):
                    for i, coordendas in enumerate(self.all_coordinates[0:(self.number_agents // 2)]):
                        is_near = SoccerEnv.is_near(player_position, coordendas, 15.0)
                        if is_near and (self.all_coordinates[-1] == coordendas).all():
                            self.all_coordinates[-1] = player_position
                            print(player_name,f"Roubou a bola de {self.player_names[i]}")
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
        new_position = old_position + np.array(t_action_direction) * 2.5 * foward_direction
        new_position = np.clip(new_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))

        if team == TEAM_LEFT_NAME:
            angle_center = np.arctan2(t_action_direction[1], t_action_direction[0])
            min_distance = float('inf')
            nearest_player_index = -1

            for i, coordenadas in enumerate(self.all_coordinates[:(self.number_agents // 2)]):
                is_near = SoccerEnv.is_near(new_position, coordenadas, 15.0)
                if is_near:
                    angle_to_player = np.arctan2(coordenadas[1] - new_position[1], coordenadas[0] - new_position[0])
                    angle_difference = self.angle_difference(angle_to_player, angle_center)
                    
                    if angle_difference < np.pi / 4:  # Considera jogadores dentro de 45 graus do ângulo central
                        distance = np.linalg.norm(new_position - coordenadas)

                        if distance < min_distance:
                            min_distance = distance
                            nearest_player_index = i

            if nearest_player_index != -1:
                self.all_coordinates[-1] = self.all_coordinates[nearest_player_index]
            else:
                self.all_coordinates[-1] = new_position
        else:
            angle_center = np.arctan2(t_action_direction[1], t_action_direction[0])
            min_distance = float('inf')
            nearest_player_index = -1

            for i, coordenadas in enumerate(self.all_coordinates[(self.number_agents // 2):self.number_agents]):
                is_near = SoccerEnv.is_near(new_position, coordenadas, 15.0)
                if is_near:
                    angle_to_player = np.arctan2(coordenadas[1] - new_position[1], coordenadas[0] - new_position[0])
                    angle_difference = self.angle_difference(angle_to_player, angle_center)
                    
                    if angle_difference < np.pi / 4:  # Considera jogadores dentro de 45 graus do ângulo central
                        distance = np.linalg.norm(new_position - coordenadas)

                        if distance < min_distance:
                            min_distance = distance
                            nearest_player_index = i + (self.number_agents // 2)  # Ajusta o índice para o time oposto

            if nearest_player_index != -1:
                self.all_coordinates[-1] = self.all_coordinates[nearest_player_index]
            else:
                self.all_coordinates[-1] = new_position


    def __kick_ball(self, all_coordinates_index: int, foward_direction: np.array):
        old_position = self.all_coordinates[-1].copy()
        new_position = old_position + 1.5 * self.player_directions[all_coordinates_index] * foward_direction
        new_position = np.clip(new_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))
        self.all_coordinates[-1] = new_position


    def goalkeeper(self, team):
        """
        Makes the goalkeeper of the playing team walk randomly on the y-axis
        """
        if team == TEAM_LEFT_NAME:
            old_position = self.all_coordinates[0].copy()

            # Limite superior e inferior da linha do gol
            upper_limit = FIELD_HEIGHT / 2 + self.outer_goal_height / 2
            lower_limit = FIELD_HEIGHT / 2 - self.outer_goal_height / 2

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
            old_position = self.all_coordinates[(self.number_agents // 2)].copy()

            # Limite superior e inferior da linha do gol
            upper_limit = FIELD_HEIGHT / 2 + self.outer_goal_height / 2
            lower_limit = FIELD_HEIGHT / 2 - self.outer_goal_height / 2
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
            self.all_coordinates[(self.number_agents // 2)] = new_position


    def defend_position(self, player_position: np.array, team: str, player_name: str):
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

