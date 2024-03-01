from typing import Dict, Optional, Union, Any, SupportsFloat
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers as pettingzoo_wrappers
from gymnasium.core import RenderFrame
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding
from env.modules.action.abstract_action_translator import ActionTranslator
from env.modules.action.discrete_action_translator import DiscreteActionTranslator
from env.modules.player_selector import PlayerSelector
from env.modules.draw.field_drawer import FieldDrawer
from env.modules.observation.image_observation_builder import ImageObservationBuilder
from env.modules.observation.dict_observation_builder import DictObservationBuilder
import numpy as np
import itertools
import functools

from env.constants import \
    FIELD_WIDTH, FIELD_HEIGHT, POINTS, \
    PLAYER_VELOCITY, BALL_VELOCITY, BALL_SLOW_CONSTANT, \
    TEAM_LEFT_NAME, TEAM_RIGHT_NAME, \
    MID_FIELD_X, MID_FIELD_Y, \
    GOAL_SIZE, TOP_GOAL_Y, BOTTOM_GOAL_Y, CENTER_GOAL_Y, \
    OUTER_GOAL_HEIGHT, OUTER_GOAL_WIDTH, INNER_GOAL_HEIGHT, INNER_GOAL_WIDTH, \
    YELLOW, BRIGHT_PURPLE


# https://pettingzoo.farama.org/content/environment_creation/ ##################
def make_raw_env(params_dict: dict):
    return SoccerEnv(**params_dict)


def make_wrapped_env(params_dict: dict):
    env = make_raw_env(params_dict)
    env = pettingzoo_wrappers.AssertOutOfBoundsWrapper(env)
    env = pettingzoo_wrappers.OrderEnforcingWrapper(env)
    return env

################################################################################

class SoccerEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4, # Used to define pygame clock tick rate
        "name": "Soccer-v0",
        "is_parallelizable": True
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
                 first_player_index = 0, # if use_kickoff_phase == True, this index selects the player to be controled before kickoff
                 skip_kickoff = True,
                 verbose = False,
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

        self.target_score = target_score
        self.__initialize_players(num_agents, left_start, control_goalkeeper)
        self.__initialize_action_translator(action_format)
        self.__initialize_render_function(render_mode, observation_format, color_option)
        self.__initialize_observation_builder(observation_format)
        self.observation_type = observation_format
        self.color_option = color_option
        self.first_player_index = first_player_index
        self.skip_kickoff = skip_kickoff
        self.seed() # Set random number generator and seed. MARL codebase expects this.
        self.reward_range = "onde que isso ta sendo chamado"

        if self.skip_kickoff:
            self.is_before_kickoff = False
        else:
            self.is_before_kickoff = True

        self.left_team_score = 0
        self.right_team_score = 0
        self.sparse_net_score_reward = sparse_net_score_reward
        self.ball_posession_reward = ball_posession_reward
        self.ball_posession = None
        self.last_ball_posession = None
        self.pygame_window = None
        self.pygame_clock = None
        self.verbose = verbose

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
                "ball": spaces.Box(
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
        [5] - Atualiza ball_posession de acordo com a posição inicial do time
        """
        # If needed -> References to see reset() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
    
        # [1] - Initialize PlayerSelector
        self.player_selector = PlayerSelector(
            self.agents, 
            self.left_start, 
            self.first_player_index, 
            self.skip_kickoff
        )
        player_name, _1, _2, team, _3 = self.player_selector.get_info()
        self.agent_selection = player_name # O nome deve ser agent_selection devido a compatibilidade com wrappers do PettingZoo
        
        # [2] - Build observation data structure
        self.observation = self.observation_builder.build_observation(
            self.all_coordinates[TEAM_LEFT_NAME].copy(),
            self.all_coordinates[TEAM_RIGHT_NAME].copy(),
            self.all_coordinates["ball"].copy(),
            self.all_coordinates["left_goalkeeper"].copy(),
            self.all_coordinates["right_goalkeeper"].copy(),
            team == TEAM_RIGHT_NAME,
            self.colors
        )
        
        # [3] - Define for all agents rewards, cumulative rewards, etc.
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # [4] - Define global state variables
        if self.skip_kickoff:
            self.is_before_kickoff = False
        else:
            self.is_before_kickoff = True

        self.left_team_score = 0
        self.right_team_score = 0
        self.ball_posession = None
        self.last_ball_posession = None

        # [5] - Atualiza ball_posession de acordo com a posição inicial do time
        team_coordinates = self.all_coordinates[team]
        ball_position = self.all_coordinates["ball"]
        for player_coord in team_coordinates:
            is_near = SoccerEnv.is_near_1(ball_position, player_coord, 0.3)
            if is_near:
                self.all_coordinates["ball"] = player_coord
                self.ball_posession = team
                break

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
        [4] - Apply action to selected player
        [5] - Check kickoff logic
        [6] - Check goal logic
        [7] - Change selected player
        [8] - Calculate reward
        [9] - Update observation
        [10] - Render
        [11] - Render
        """
        # If needed -> References to see step() method:
        # https://pettingzoo.farama.org/content/environment_creation/
        # https://www.gymlibrary.dev/content/environment_creation/
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py#L226
        def _step(action: tuple[int,int]):
            # [1] - Translate action
            t_action = self.action_translator.translate_action(action)
            
            # [2] - Get select player to make action and context info
            player_name, foward_direction, _, team, other_team = self.player_selector.get_info()
            dict_index, array_index = self.player_name_to_index[player_name]

            # [3] - Removes, if defending, the current player of the defending list
            self.clean_defended_positions(player_name, team)

            # [4] - Apply action
            penalty = self.actions(action, t_action.direction)
            if not self.control_goalkeeper: 
                self.auto_move_goalkeeper(team)
            player_position = self.all_coordinates[team][dict_index]
            ball_position = self.all_coordinates["ball"]

            # [5] - Check kickoff logic
            has_moved = action[1] == 0
            if (not self.skip_kickoff
                and self.is_before_kickoff
                and has_moved
                and SoccerEnv.is_near_1(player_position, ball_position, 1.2)):
                    
                    # Autograb the ball if near enough and 
                    # no player is in the same pos of the ball 
                    self.all_coordinates["ball"] = player_position
                    self.ball_posession = team
                    self.is_before_kickoff = False
                    
                    self.player_selector.playing_rotation()
                    if self.verbose:
                        print("@@@@@@@@ Aconteceu kickoff | Entrou na fase de playing rotation @@@@@@@@")

            # [6] - Check goal logic
            # Remember: y axis grows down ([0,1] points down), hence "> top" & "< botton"
            left_team_goal = None
            if (ball_position[0] == 0
                and ball_position[1] > TOP_GOAL_Y
                and ball_position[1] < BOTTOM_GOAL_Y):
                
                if self.verbose:
                    print(f"> GOL {TEAM_RIGHT_NAME} GOL <")
                self.right_team_score += 1
                left_team_goal = False
                self.player_selector.kickoff_rotation(TEAM_LEFT_NAME)
                if self.skip_kickoff:
                    self.player_selector.playing_rotation()
                self.is_before_kickoff = not self.skip_kickoff # False if self.skip_kickoff else True

                # Reset position
                self.all_coordinates[TEAM_LEFT_NAME] = self.start_positions_left_kickoff
                self.all_coordinates[TEAM_RIGHT_NAME] = self.start_positions_right_kickoff
                self.all_coordinates["ball"] = self.start_ball_position
                self.all_coordinates["left_goalkeeper"] = self.start_left_goalkeeper_position
                self.all_coordinates["right_goalkeeper"] = self.start_right_goalkeeper_position
                self.player_directions = self.start_directions

                team_coordinates = self.all_coordinates[TEAM_LEFT_NAME]
                ball_position = self.all_coordinates["ball"]
                for player_coord in team_coordinates:
                    is_near = SoccerEnv.is_near_1(ball_position, player_coord, 0.3)
                    if is_near:
                        self.all_coordinates["ball"] = player_coord
                        self.ball_posession = team
                        break

            elif (ball_position[0] == FIELD_WIDTH
                and ball_position[1] > TOP_GOAL_Y
                and ball_position[1] < BOTTOM_GOAL_Y):
                
                if self.verbose:
                    print(f"> GOL {TEAM_LEFT_NAME} GOL <")
                self.left_team_score += 1
                left_team_goal = True
                self.player_selector.kickoff_rotation(TEAM_RIGHT_NAME)
                if self.skip_kickoff:
                    self.player_selector.playing_rotation()
                self.is_before_kickoff = not self.skip_kickoff # False if self.skip_kickoff else True

                # Reset position
                self.all_coordinates[TEAM_LEFT_NAME] = self.start_positions_left_kickoff
                self.all_coordinates[TEAM_RIGHT_NAME] = self.start_positions_right_kickoff
                self.all_coordinates["ball"] = self.start_ball_position
                self.all_coordinates["left_goalkeeper"] = self.start_left_goalkeeper_position
                self.all_coordinates["right_goalkeeper"] = self.start_right_goalkeeper_position
                self.player_directions = self.start_directions

                team_coordinates = self.all_coordinates[TEAM_RIGHT_NAME]
                ball_position = self.all_coordinates["ball"]
                for player_coord in team_coordinates:
                    is_near = SoccerEnv.is_near_1(ball_position, player_coord, 0.3)
                    if is_near:
                        self.all_coordinates["ball"] = player_coord
                        self.ball_posession = team
                        break

            # [7] - Change selected player
            self.player_selector.next_player()
            player_name, _1, _2, _3, _4 = self.player_selector.get_info()
            self.agent_selection = player_name

            # [8] - Calculate reward
            reward = penalty
        
            if self.ball_posession_reward:
                if SoccerEnv.is_in_any_array(ball_position, self.all_coordinates[team]):
                    reward += 0.1

            if not self.sparse_net_score_reward:
                if team == TEAM_LEFT_NAME:
                    reward += self.left_team_score - self.right_team_score
                else:
                    reward += self.right_team_score - self.left_team_score

            elif self.sparse_net_score_reward:
                # XOR check if (team == TEAM_LEFT_NAME and not left_team_goal) or (team != TEAM_LEFT_NAME and left_team_goal):
                if not(team == TEAM_LEFT_NAME ^ left_team_goal):
                    reward += 1
                else:
                    reward -= 1

            if self.last_ball_posession is not None and self.last_ball_posession != self.ball_posession:
                if team == self.ball_posession:
                    reward += 0.01
                else:
                    reward -= 0.01

            # Update player reward
            self.rewards[player_name] = reward
            self._cumulative_rewards[player_name] += reward

            # [9] - Update observation
            self.observation = self.observation_builder.build_observation(
                self.all_coordinates[TEAM_LEFT_NAME].copy(),
                self.all_coordinates[TEAM_RIGHT_NAME].copy(),
                self.all_coordinates["ball"].copy(),
                self.all_coordinates["left_goalkeeper"].copy(),
                self.all_coordinates["right_goalkeeper"].copy(),
                team == TEAM_RIGHT_NAME,
                self.colors
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

            # [10] - Render
            if self.render_mode == 'human':
                self.render()
            # else:
            #     self.debug_render()

            # [11] - Check end of episode
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
    
        if isinstance(action, list):
            cum_reward = []
            for a in action:
                last_observation, r, terminated, truncated, info = _step(a)
                cum_reward.append(r)
                return last_observation, r, terminated, truncated, info
        else:
            return _step(action)
    

    def __initialize_players(self, 
                             num_agents: int, 
                             left_start: bool, 
                             control_goalkeeper: bool
                             ) -> None:

        # ################ #####################################################
        # Define functions #####################################################
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
                flip: bool = False
                )-> list[np.ndarray[np.float32]]:
            """
            n(int): Quantidade de coordenadas geradas
            left_start(bool): Caso True, faz o segundo jogador iniciar muito perto da bola.
            flip(bool): Caso True, Espelha as posições x para o outro lado do campo.
            """
            t = np.float32
            x_near = 1.0 if left_start else 0.8
            
            # 10 positions to pick. if n > 10, then these will repeat
            # Players are positioned with some relative spacing
            # Note: Adjust the coordinates based on preferred formation nuances
            predefined_positions = [
                np.array([MID_FIELD_X * 0.8, MID_FIELD_Y * 0.8 ], dtype=t), # [0] Forward up
                np.array([MID_FIELD_X*x_near, MID_FIELD_Y      ], dtype=t), # [1] Forward middle, slightly ahead for kickoff
                np.array([MID_FIELD_X * 0.8, MID_FIELD_Y * 1.2 ], dtype=t), # [2] Forward down
                np.array([MID_FIELD_X * 0.3, MID_FIELD_Y * 0.75], dtype=t), # [3] Defender up
                np.array([MID_FIELD_X * 0.3, MID_FIELD_Y * 1.25], dtype=t), # [4] Defender down
                np.array([MID_FIELD_X * 0.4, MID_FIELD_Y * 0.5 ], dtype=t), # [5] Defender up
                np.array([MID_FIELD_X * 0.4, MID_FIELD_Y * 1.5 ], dtype=t), # [6] Defender down
                np.array([MID_FIELD_X * 0.6, MID_FIELD_Y * 0.6 ], dtype=t), # [7] Midfielder up
                np.array([MID_FIELD_X * 0.6, MID_FIELD_Y       ], dtype=t), # [8] Midfielder middle
                np.array([MID_FIELD_X * 0.6, MID_FIELD_Y * 1.4 ], dtype=t), # [9] Midfielder down
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
        

        def goalkeeper_coordinates(use_noise: bool = True)-> list[np.ndarray[np.float32]]:
            t = np.float32

            if use_noise:
                noise = np.random.uniform(FIELD_HEIGHT/2 + GOAL_SIZE/2, FIELD_HEIGHT/2 - GOAL_SIZE/2)
                goalkeeper_pos = np.array([0.1, noise], dtype=t)
            else:
                goalkeeper_pos = np.array([0.1, MID_FIELD_Y], dtype=t)

            left_goalkeeper_pos = goalkeeper_pos
            right_goalkeeper_pos = left_goalkeeper_pos.copy()
            right_goalkeeper_pos[0] = FIELD_WIDTH - goalkeeper_pos[0]

            return [left_goalkeeper_pos, right_goalkeeper_pos]
        
        # #################### #####################################################
        # Define env variables #####################################################

        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.half_possible_agents = num_agents // 2

        l_goalkeeper_index = (num_agents // 2) -1
        r_goalkeeper_index = -1
        self.possible_agents[l_goalkeeper_index] = self.possible_agents[l_goalkeeper_index].replace("player", "goalkeeper")
        self.possible_agents[r_goalkeeper_index] = self.possible_agents[r_goalkeeper_index].replace("player", "goalkeeper")

        self.agents = self.possible_agents.copy()
        self.half_number_agents = num_agents // 2
        self.team_agents = {
            TEAM_LEFT_NAME: self.agents[:self.half_number_agents],
            TEAM_RIGHT_NAME: self.agents[self.half_number_agents:]
        }
        self.number_agents = num_agents
        self.n_agents = num_agents # n_agents is for compatibiliy with MARL codebase
        self.left_start = left_start
        self.control_goalkeeper = control_goalkeeper

        # Remove 2 agents from controled agents
        if not control_goalkeeper:
            # Must remove rightmost index first
            self.agents.pop(r_goalkeeper_index) # remove right goalkeeper
            self.agents.pop(l_goalkeeper_index) # remove left goalkeeper

            self.team_agents[TEAM_LEFT_NAME].pop(-1) # remove left goalkeeper
            self.team_agents[TEAM_RIGHT_NAME].pop(-1) # remove right goalkeeper

            # Decrement
            self.number_agents += -2
            self.n_agents += -2
            self.half_number_agents += -1

        # First n players will be left side and last n will be right side
        left_team_coordinates = deterministic_coordinate_generator(self.half_number_agents, left_start)
        right_team_coordinates = deterministic_coordinate_generator(self.half_number_agents, not left_start, flip=True)
        left_goalkeeper, right_goalkeeper = goalkeeper_coordinates()
        ball_position = np.array(POINTS["center"], dtype=np.float32)
        
        # self.all_coordinates = left_goalkeeper + \
        #                        left_team_coordinates + \
        #                         right_goal_keeper + \
        #                         right_team_coordinates + \
        #                        [ball_position]
        
        self.all_coordinates = {
            TEAM_LEFT_NAME:  left_team_coordinates + [left_goalkeeper],
            TEAM_RIGHT_NAME: right_team_coordinates + [right_goalkeeper],
            "ball": ball_position,
            "left_goalkeeper": [],
            "right_goalkeeper": []
        }
        if not control_goalkeeper:
            # Must remove rightmost index first
            self.all_coordinates[TEAM_RIGHT_NAME].pop(r_goalkeeper_index)
            self.all_coordinates[TEAM_LEFT_NAME].pop(l_goalkeeper_index)

            self.all_coordinates["left_goalkeeper"] = left_goalkeeper
            self.all_coordinates["right_goalkeeper"] = right_goalkeeper
            

        
        # mapping agent_name to all indexes used in the code
        indexes = list(zip(
            list(range(self.half_number_agents)) * 2, # Use in -> all_coordinates[team][index]
            list(range(self.number_agents))           # Use in -> self.player_directions[index]
        ))
        self.player_name_to_index = dict( 
            zip(self.agents, indexes)
        )

        # Initialize player directions and defended positions.
        left_foward = (1, 0)
        right_foward = (-1, 0)
        self.player_directions = np.array(
            [left_foward]  * (self.half_number_agents) + \
            [right_foward] * (self.half_number_agents),
            dtype=np.uint8
        )
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
        self.start_positions_left_kickoff = self.all_coordinates[TEAM_LEFT_NAME].copy()
        self.start_positions_right_kickoff = self.all_coordinates[TEAM_RIGHT_NAME].copy()
        self.start_ball_position = ball_position.copy()
        self.start_left_goalkeeper_position = self.all_coordinates["left_goalkeeper"].copy()
        self.start_right_goalkeeper_position = self.all_coordinates["right_goalkeeper"].copy()
        self.start_directions = self.player_directions.copy()

        # print('ALL LEFT COORDINATES:', self.all_coordinates[TEAM_LEFT_NAME])
        # print('ALL RIGHT COORDINATES:', self.all_coordinates[TEAM_RIGHT_NAME])


    def __initialize_action_translator(self, action_format):
        if action_format == 'discrete':
            self.action_translator = DiscreteActionTranslator()
        else:
            raise Exception("Action format is not recognized")
        

    def __initialize_render_function(self, 
                                     render_mode: str = "human", 
                                     observation_format: str = "dict", 
                                     color_option: int = 0
                                     ) -> None:
        
        # Site to visualize color schemas -> https://www.realtimecolors.com/?colors=050316-0f5415-24972e-c528f0-f9e636&fonts=Poppins-Poppins
        color_schemas: list[dict[str, any]] = [
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
        self.colors = color_schemas[color_option]


        def human_render():
            all_positions = self.all_coordinates[TEAM_LEFT_NAME]       + \
                            [self.all_coordinates["left_goalkeeper"]]  + \
                            self.all_coordinates[TEAM_RIGHT_NAME]      + \
                            [self.all_coordinates["right_goalkeeper"]]
            
            field_image = self.field_drawer.draw_field(
                all_positions, 
                self.all_coordinates["ball"],
                **self.colors
            )

            field_image = np.array(field_image).transpose((1, 0, 2))

            """Fetch the last frame from the base environment and render it to the screen."""
            try:
                import pygame
            except ImportError as e:
                raise DependencyNotInstalled(
                    "pygame is not installed, run `pip install pygame`"
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
            all_positions = self.all_coordinates[TEAM_LEFT_NAME]       + \
                            [self.all_coordinates["left_goalkeeper"]]  + \
                            self.all_coordinates[TEAM_RIGHT_NAME]      + \
                            [self.all_coordinates["right_goalkeeper"]]
            field_image = self.field_drawer.draw_field(
                all_positions, 
                self.all_coordinates["ball"],
                **self.colors
            )
            return field_image
                

        render_functions = {
            "human": human_render,
            "rgb_array": rgb_array_render
        }
        # self.debug_render =  render_functions["human"]
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
    def is_in_any_array(some_position: np.ndarray, 
                        position_matrix: Union[list, np.ndarray]
                        ) -> bool:
        """
        Verifica se some_position é igual a pelo menos um dos arrays na pos_matrix.

        - some_position(np.ndarray): O array que você deseja verificar.
        - position_matrix(np.ndarray): Uma matriz contendo arrays de 2 elementos para comparação.
        
        Returns True if some_position is equal to at least one array in position_matrix, False otherwise.
        """
        return np.any(np.all(some_position == position_matrix, axis=1))
    

    @staticmethod
    def is_near_1(pos_1: np.array, 
                  pos_2: np.array, 
                  threshold: float = 5.0
                  ) -> bool:

        euclidean_distance = np.linalg.norm(pos_1 - pos_2)

        return euclidean_distance <= threshold
    

    @staticmethod
    def is_near_2(pos_1: np.array, 
                  pos_list: Union[list, np.array], 
                  threshold: float = 5.0
                  ) -> bool:
        euclidean_distance = np.linalg.norm(pos_1 - pos_list, axis=1)

        return np.any(euclidean_distance <= threshold)


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
    
    
    # na tupla de ação, o primeiro elemento é a direção e o segundo é a ação, nas ações 0 é andar, 1 é roubar bola, 2 é passe, 3 é chute e 4 é proteger a bola
    # Falta conferir os valor fixos que estou utilizando, para ver de qual distancia pode roubar a bola, 
    # a distancia do passe e também a distancia que o jogador tem que estar da bola para ele receber o passe, 
    # também falta conferir a distancia do chute
    # retorna uma penalidade por executar ações indevidas, ou zero
    def actions(self, action: tuple[int,int], action_direction: np.array) -> float:
        
        # Get select player to make action and context info
        player_name, foward_direction, _, team, other_team = self.player_selector.get_info()
        dict_index, array_index = self.player_name_to_index[player_name]

        player_position = self.all_coordinates[team][dict_index]
        ball_position = self.all_coordinates["ball"]

        is_near = SoccerEnv.is_near_1(player_position, ball_position, 3.0)
        has_the_ball = (player_position == ball_position).all()

        penalty = 0
        # player para enquanto executa ações
        if action[1] == 0:
            self.__move_player(action_direction)
        elif action[1] == 1:
            penalty = self.__steal_ball_action()
        # Ação de passe inteligente implementado, conferindo o jogador mais proximo da localização em que a bola pararia apos o passe.
        elif action[1] == 2 and has_the_ball:
            self.__pass_ball(action_direction)
        elif action[1] == 3 and has_the_ball:
            self.__kick_ball(action_direction, action[0])
        elif action[1] == 4:
            self.__defend_position()
        
        if not is_near and action[1] >= 2:
            action_name = {
                2: "pass_ball",
                3: "kick_ball",
                4: "defend_position"
            }
            if self.verbose:
                print(f"{team} {player_name} lose reward trying to {action_name[action[1]]} | Indexes({dict_index}, {array_index}, {self.player_selector._index})")
            penalty = -0.01

        return penalty


    def __steal_ball_action(self) -> None:
        
        # Get select player to make action and context info
        player_name, foward_direction, _, team, other_team = self.player_selector.get_info()
        dict_index, array_index = self.player_name_to_index[player_name]

        player_position = self.all_coordinates[team][dict_index]
        ball_position = self.all_coordinates["ball"]
        defended_positions = self.defend_positions[other_team]["positions"]

        if len(defended_positions) > 0 and \
           SoccerEnv.is_near_2(player_position, defended_positions, 3.0):
            defense_bonus = +0.1
        else: 
            defense_bonus = 0

        if np.random.rand() < 0.5 + defense_bonus:
            steal_ball = True
            if self.verbose:
                print(f"{team} {player_name} tentou roubar bola Indexes({dict_index}, {array_index}, {self.player_selector._index})")
        else:
            steal_ball = False
            if self.verbose:
                print(f"{team} {player_name} deu azar e falhou ao tentar roubar bola Indexes({dict_index}, {array_index}, {self.player_selector._index})") 

        penalty = 0
        if steal_ball:
            if SoccerEnv.is_in_any_array(ball_position, self.all_coordinates[other_team]):
                for i, other_player_coord in enumerate(self.all_coordinates[other_team]):
                    is_near = SoccerEnv.is_near_1(player_position, other_player_coord, 15.0)
                    if is_near and (ball_position == other_player_coord).all():
                        self.all_coordinates["ball"] = player_position
                        if self.verbose:
                            if team == TEAM_LEFT_NAME:
                                other_player_name = self.agents[i + self.half_number_agents]
                            else:
                                other_player_name = self.agents[i]
                            print(f"{player_name} roubou a bola de {other_player_name}")
                        self.last_ball_posession = self.ball_posession
                        self.ball_posession = team

                        penalty = +0.01 # Sobrescreve -0.01 caso entre no if
                        break
                    else:
                        penalty = -0.01
            else:
                penalty = -0.01

        return penalty


    def __move_player(self, action_direction: np.ndarray) -> None:
        
        # Get select player to make action and context info
        player_name, foward_direction, _, team, other_team = self.player_selector.get_info()
        dict_index, array_index = self.player_name_to_index[player_name]

        ball_position = self.all_coordinates["ball"]

        # Calcula nova posição
        old_position = self.all_coordinates[team][dict_index].copy()
        new_position = old_position + action_direction * PLAYER_VELOCITY * foward_direction
        new_position = np.clip(new_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))

        # Faz a posição da bola acompanhar o jogador
        if self.ball_posession == team and (old_position == ball_position).all():
            self.all_coordinates["ball"] = new_position

        # Ao chegar na posição nova, pega a bola automaticamente caso a bola esteja livre
        if self.ball_posession is not other_team and \
           SoccerEnv.is_near_1(new_position, ball_position, 0.4):
            self.all_coordinates["ball"] = new_position
            self.last_ball_posession = self.ball_posession
            self.ball_posession = team

        # Atualiza posição do jogador
        self.all_coordinates[team][dict_index] = new_position
        self.player_directions[array_index] = action_direction

        if self.verbose:
            print(f"{team} {player_name} move from ({old_position}) to ({new_position}) | Indexes({dict_index}, {array_index}, {self.player_selector._index})")


    def __pass_ball(self, action_direction: np.ndarray) -> None:
        
        # Get select player to make action and context info
        player_name, foward_direction, _, team, other_team = self.player_selector.get_info()
        dict_index, array_index = self.player_name_to_index[player_name]

        old_position = self.all_coordinates["ball"].copy()
        new_position = old_position + np.array(action_direction) * BALL_VELOCITY * foward_direction
        if self.verbose:
            print("forward_direction", foward_direction)
            print("t_action_direction", action_direction)
        new_position = np.clip(new_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))

        # Calcula o ângulo entre a direção do passe e a posição atual do jogador
        angle_center = np.arctan2(action_direction[1], action_direction[0]) 
        angle_center += np.arctan2(foward_direction[1], foward_direction[0]) # Resulta em 0 (0º) ou pi (180º)
        if self.verbose:
            print("Angle center", angle_center)

        # if team == TEAM_LEFT_NAME:
        #     players_range = range(self.half_number_agents)
        # else:
        #     players_range = range(self.half_number_agents, self.number_agents)

        min_angle_difference = float('inf')
        nearest_player_index = -1

        for dict_i, player_coord in enumerate(self.all_coordinates[team]):

            is_near = SoccerEnv.is_near_1(old_position, player_coord, 15.0)

            if is_near:
                ball_to_player_vector = player_coord - old_position # zero quando player_coord é o jogador com a bola (current_player)
                player_distance = np.linalg.norm(ball_to_player_vector)
                #print("ball_to_player_vector", ball_to_player_vector)
                #print("player_distance", player_distance)
                #print("player_coord", player_coord)

                if player_distance != 0:  # Evita divisão por zero
                    ball_to_player_vector /= player_distance  # Normaliza o vetor para ter comprimento 1
                    dot_product = np.dot(action_direction, ball_to_player_vector)
                    #print("dot_product", dot_product)
                    if dot_product >= 0:  # Verifica se o jogador está na direção do passe
                    # Calcula o ângulo entre o vetor posição do jogador e a direção do passe
                        angle_to_player = np.arctan2(ball_to_player_vector[1], ball_to_player_vector[0])
                        angle_difference = np.abs(self.angle_difference(angle_to_player, angle_center))
                        #print("angle_difference", angle_difference)
                        if angle_difference < min_angle_difference or (angle_difference == min_angle_difference and player_distance < min_distance_to_passer):
                            min_angle_difference = angle_difference
                            nearest_player_index = dict_i
                            min_distance_to_passer = player_distance
    
        if nearest_player_index != -1:
            self.all_coordinates["ball"] = self.all_coordinates[team][nearest_player_index]
        else:
            self.all_coordinates["ball"] = new_position
            self.last_ball_posession = self.ball_posession
            self.ball_posession = None

        if self.verbose:
            print(f"{team} {player_name} passed the ball | Indexes({dict_index}, {array_index}, {self.player_selector._index})")


    def __kick_ball(self, 
                    action_direction: np.ndarray, 
                    action_direction_index: int
                    ) -> None:
        
        # Get select player to make action and context info
        player_name, foward_direction, _, team, other_team = self.player_selector.get_info()
        dict_index, array_index = self.player_name_to_index[player_name]
        
        player_position: np.ndarray[np.float32] = self.all_coordinates[team][dict_index]
        old_ball_position = self.all_coordinates["ball"].copy()

        if action_direction_index in [1, 5, 7]:
            # Escolheu chutar para trás
            final_direction = action_direction * foward_direction
        elif action_direction_index == 8:
            # Sem direção escolhida, então chuta para direção atual do jogador
            final_direction = self.player_directions[array_index] * foward_direction
            final_direction = SoccerEnv.normalize_direction(final_direction)
        else:

            # Team side adjustments
            if team == TEAM_LEFT_NAME:
                tiangle_x_lenght = FIELD_WIDTH - player_position[0]
                center_goal_x = FIELD_WIDTH
            else:
                tiangle_x_lenght = player_position[0]
                center_goal_x = 0
                
            # change y to make calculations considering y growing up instead of down
            action_direction[1] *= -1 

            # [1] - Rotacionar t_action_direction em + e - 45 graus
            rotation_angle = 45
            upper_vector = SoccerEnv.rotate_vector(action_direction, angle_degrees= rotation_angle)
            lower_vector = SoccerEnv.rotate_vector(action_direction, angle_degrees= -rotation_angle)

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
            # Other way of computing bottom_y_intersection
            # bottom_y_intersection = (player_position[1] - upper_y_intersection) + player_position[1]

            # TODO make an optimized version of above code considering only [+90, +45, 0, -45, -90] angles
            # This commented code is an optimized version of +-45 degree
            # top_y_intersection = player_position[1] - player_position[0]
            # bottom_y_intersection = player_position[1] + player_position[0]

            can_kick_to_goal = \
                (upper_vector[0] > 0 and CENTER_GOAL_Y >= top_y_intersection) \
                or \
                (lower_vector[0] > 0 and CENTER_GOAL_Y <= bottom_y_intersection)
            
            if self.verbose:
                print(f"player_position: {player_position}")
                print(f"horizontal_vector: {horizontal_vector}")
                print(f"t_action_direction: {action_direction}")
                print(f"upper_vector: {upper_vector}")
                print(f"upper_angle_degree: {np.degrees(upper_angle_radians)}")
                print(f"lower_vector: {lower_vector}")
                print(f"lower_angle_degree: {np.degrees(lower_angle_radians)}")
                print(f"top_y_intersection: {top_y_intersection}")
                print(f"bottom_y_intersection: {bottom_y_intersection}")
                print(f"CENTER_GOAL_Y: {CENTER_GOAL_Y}")
                print(f"pode chutar ? {can_kick_to_goal}")

            if can_kick_to_goal:
                center_goal_point = np.array((center_goal_x, CENTER_GOAL_Y), dtype=np.float32)
                final_direction = center_goal_point - player_position
                final_direction = SoccerEnv.normalize_direction(final_direction)
            else:
                action_direction[1] *= -1 # Revert y signal to consider y growing down
                final_direction = action_direction  * foward_direction

        # Calculate and update ball position
        new_ball_position = old_ball_position + BALL_VELOCITY * final_direction
        new_ball_position = np.clip(new_ball_position, (0, 0), (FIELD_WIDTH, FIELD_HEIGHT))
        self.all_coordinates["ball"] = new_ball_position

        # Update ball posession
        self.last_ball_posession = self.ball_posession
        self.ball_posession = None

        if self.verbose:
            print(f"{team} {player_name} kicked the ball | Indexes({dict_index}, {array_index}, {self.player_selector._index})")



    def auto_move_goalkeeper(self, team):
        """
        Makes the goalkeeper of the playing team walk randomly on the y-axis
        """
        # Limite superior e inferior da linha do gol
        upper_limit = FIELD_HEIGHT // 2 - GOAL_SIZE // 2
        lower_limit = FIELD_HEIGHT // 2 + GOAL_SIZE // 2

        # Movimentação aleatória para cima ou para baixo
        move_direction = np.random.choice([-1, 1])

        # Distância aleatória para se movimentar (você pode ajustar a amplitude conforme necessário)
        move_distance = np.random.uniform(0, 1.5)
        
        # Pega posição y do goleiro
        if team == TEAM_LEFT_NAME:
            key = "left_goalkeeper"
        else:
            key = "right_goalkeeper"
            
        y = self.all_coordinates[key][1]

        # Calculando nova posição y
        new_y = y + move_direction * move_distance
        new_y = np.clip(new_y, lower_limit, upper_limit)

        # Atualizando a posição apenas na coordenada y
        self.all_coordinates[key][1] = new_y


    def __defend_position(self) -> None:
        """
        Makes the player defend the position by adding its coordinates into a list.
        This list is used inside steal_ball action to reduce the odds of stealing.
        Each team has its list and all defended positions are cleaned after some turns.
        """
        # Get select player to make action and context info
        player_name, _, _, team, _ = self.player_selector.get_info()
        dict_index, array_index = self.player_name_to_index[player_name]

        player_position = self.all_coordinates[team][dict_index]

        self.defend_positions[team]["positions"].append(player_position)
        self.defend_positions[team]["player_names"].append(player_name)

        if self.verbose:
            print(f"{team} {player_name} is defending a position |  Indexes({dict_index}, {array_index}, {self.player_selector._index})")


    def clean_defended_positions(self, player_name: str, team: str):
        """
        Checks if the current player name is in a defended position and removes it.
        If a player takes the defend ball action, then it will be added back to the list.
        """
        defending_players_names = self.defend_positions[team]["player_names"]
        if player_name in defending_players_names:
            index = defending_players_names.index(player_name)
            self.defend_positions[team]["positions"].pop(index)
            self.defend_positions[team]["player_names"].pop(index)

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def observation(self, agent: Optional[str] = None):
        
        # retorna a posição do agente caso o nome do agente não seja None
        if agent:

            _, all_coordinates_index = self.player_name_to_index[agent]
            if all_coordinates_index < self.half_number_agents:
                team = TEAM_LEFT_NAME
            else:
                team = TEAM_RIGHT_NAME
                
            return self.observation[team][agent]
        
        return self.observation
        
    
    def observe(self, agent):
        """
        Esse observe retorna a observação completa. 
        É possível criar um wrapper para filtrar a observação por agente.
        """

        # Uncomment if need to rebuild the observation
        # dict_index, array_index = self.player_name_to_index[agent]
        # is_right_team = array_index >= self.half_number_agents
        # self.observation = self.observation_builder.build_observation(
        #     self.all_coordinates[TEAM_LEFT_NAME].copy(),
        #     self.all_coordinates[TEAM_RIGHT_NAME].copy(),
        #     self.all_coordinates["ball"].copy(),
        #     is_right_team,
        #     self.colors
        # )

        return self.observation
    

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Union[spaces.Box, Dict]:
        return self._get_observation_spaces()


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        return self.action_translator.action_space()
