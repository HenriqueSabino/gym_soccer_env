from typing import Optional
import numpy as np

from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME
TOGGLE_X_ONLY = np.array([-1, 1])

class PlayerSelector:
    def __init__(self, 
                 player_names: list[str],
                 left_start: bool,
                 kickoff_player_index = 2,
                 control_goalkeeper = False,
                 skip_kickoff = True,
                 verbose = False,
                ):
        """
        #### Params
        player_names(list[str]): list with all players names.\n
        left_start(bool): Indicates if left of right team starts.\n
        kickoff_player_index(int): index to pick from sorted list.\n
        control_goalkeeper(bool): if True, keeps the goal keeper in the seleciton.\n
        skip_kickoff(bool): if True, skips kickoff rotation in toggle.\n
        \n
        #### Warning
        kickoff_player_index must be even to start with left team \n
        odd indexes -> right_team \n
        even indexes -> left_team \n
        \n
        The sort step determines who plays first, second and so on.\n
        The sort step alternates between left and right.
        """
        assert len(player_names) % 2 == 0, \
            "Number of agents must be even. \n" + \
            "(i//2 + self.player_count//2) expression" + \
            "breaks with odd player_count. \n"
        
        if not control_goalkeeper:
            assert kickoff_player_index < len(player_names)-2, \
                f"kickoff_player_index must be from 0 to {len(player_names)-3}. \n"
        else:
            assert kickoff_player_index < len(player_names), \
                f"kickoff_player_index must be from 0 to {len(player_names)-1}. \n" 
            
            assert kickoff_player_index == 0 or kickoff_player_index == 1, \
                "First player index must NOT be 0 or 1. \n" + \
                "Index 0 and 1 gets the goalkeeper of team left and right respectively. \n" + \
                "Index greater than 1 gets a player. \n"
        
        assert (left_start and kickoff_player_index % 2 == 0) or \
            (not left_start and kickoff_player_index % 2 != 0), \
            "kickoff_player_index must be even to start with left team. \n" + \
            "odd indexes -> right_team \n" + \
            "even indexes -> left_team \n"
            
        self.player_count = len(player_names)
        
        # Sort player names in the playing rotation order of turns
        self.player_order_to_play = [
            player_names[i//2] if i % 2 == 0 
            else player_names[i//2 + self.player_count//2] 
            for i in range(self.player_count)
        ]

        if not control_goalkeeper:
            # Remove goalkeeper of rotation
            self.player_order_to_play = self.player_order_to_play[2:]
            self.player_count = len(player_names) - 2

        # _index keeps track of current player
        self._index = kickoff_player_index 

        self._current_player_name = self.player_order_to_play[self._index]
        self._control_goalkeeper = control_goalkeeper
        if left_start:
            self._x_foward_direction = np.array([1, 1])
            self._is_left_team = True
            self._currently_acting_team = TEAM_LEFT_NAME
            self._not_currently_acting_team = TEAM_RIGHT_NAME
            self._next_team_to_play_after_kickoff = TEAM_RIGHT_NAME
            self.left_team_kickoff_player_index = kickoff_player_index
            self.right_team_kickoff_player_index = kickoff_player_index + 1
        elif not left_start:
            self._x_foward_direction = np.array([-1, 1])
            self._is_left_team = False
            self._currently_acting_team = TEAM_RIGHT_NAME
            self._not_currently_acting_team = TEAM_LEFT_NAME
            self._next_team_to_play_after_kickoff = TEAM_LEFT_NAME
            self.left_team_kickoff_player_index = kickoff_player_index -1
            self.right_team_kickoff_player_index = kickoff_player_index

        self._skip_kickoff_rotation = skip_kickoff
        if skip_kickoff:
            self.selector_logic_callback = self._after_kickoff_logic_callback
            self._is_using_playing_rotation = True
        else:
            self.selector_logic_callback = self._before_kickoff_logic_callback
            self._is_using_playing_rotation = False

        self.verbose = verbose
            


    def get_info(self) -> tuple[str, bool, np.array]:
        """
        Return tuple containing \n
        [0] (str)      Current player name \n
        [1] (np.array) [+1,1] or [-1,1] indicating x foward direction \n
        [2] (bool)     Indicating if is_left_team turn \n
        [3] (str)      Name of the team currently acting \n
        [4] (str)      Name of the team NOT currently acting \n
        """
        return (
            self._current_player_name, 
            self._x_foward_direction,
            self._is_left_team,
            self._currently_acting_team,
            self._not_currently_acting_team
        )


    def next_player(self) -> str:
        """
        Pass the turn to next player and updates the internal
        _current_player_name, _is_left_team and _direction variables.
        Returns _current_player_name.
        """
        self.selector_logic_callback()
               
        return self._current_player_name


    def _toggle_side(self):
        self._x_foward_direction = self._x_foward_direction * TOGGLE_X_ONLY
        self._is_left_team ^= True # is_left_side XOR True
        temp = self._currently_acting_team
        self._currently_acting_team = self._not_currently_acting_team
        self._not_currently_acting_team = temp
        if self.verbose:
            print("Trocou time")


    # Uncomment if you need to use
    # def toggle_rotation(self, team_to_play: Optional[str] = None):

    #     # toggle to same rotation when _skip_kickoff_rotation is True
    #     if self._skip_kickoff_rotation:
    #         self.kickoff_rotation(team_to_play)
    #         self.playing_rotation()
    #     else:
    #         if self._is_using_playing_rotation:
    #             self.kickoff_rotation(team_to_play)
    #         else:
    #             self.playing_rotation()


    def playing_rotation(self):
        """
        #### Called after kickoff
        The logic to select players change after kickoff.\n
        Change selector logic callback to pick next player considering playing stage.
        """
        self.selector_logic_callback = self._after_kickoff_logic_callback
        self._is_using_playing_rotation = True

        if self._next_team_to_play_after_kickoff == TEAM_LEFT_NAME:
            self._index = 0 # Start moving the left team index 0 player
        else:
            self._index = 1 # Start moving the right team index 1 player
        
        self._index -= 1 # Remove the +1 in the next selection


    def kickoff_rotation(self, team_to_play: str):
        """
        #### Called after goal scored
        The logic to select players change after goal.\n
        Change selector logic callback to pick next player considering kickoff stage.\n
        Needs to recive team_to_play because this class doesn't know who has scored a goal.
        """
        self.selector_logic_callback = self._before_kickoff_logic_callback
        self._is_using_playing_rotation = False

        if team_to_play == TEAM_LEFT_NAME:
            self._index = self.left_team_kickoff_player_index
            self._x_foward_direction = np.array([1, 1])
            self._is_left_team = True
            self._currently_acting_team = TEAM_LEFT_NAME
            self._not_currently_acting_team = TEAM_RIGHT_NAME
            self._next_team_to_play_after_kickoff = TEAM_RIGHT_NAME
        elif team_to_play == TEAM_RIGHT_NAME:
            self._index = self.right_team_kickoff_player_index
            self._x_foward_direction = np.array([-1, 1])
            self._is_left_team = False
            self._currently_acting_team = TEAM_RIGHT_NAME
            self._not_currently_acting_team = TEAM_LEFT_NAME
            self._next_team_to_play_after_kickoff = TEAM_LEFT_NAME

    
    def _before_kickoff_logic_callback(self):
        # Uncomment the line below if all players of the left team should play before kickoff
        # Remember to self._index -= 2 inside kickoff_rotation to counter the first +2
        # self._index = (self._index + 2) % self.player_count
        self._current_player_name = self.player_order_to_play[self._index]


    def _after_kickoff_logic_callback(self):
        self._index = (self._index + 1) % self.player_count
        self._current_player_name = self.player_order_to_play[self._index]
        self._toggle_side()
