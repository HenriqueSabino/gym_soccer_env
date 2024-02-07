import numpy as np

from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME
TOGGLE_X_ONLY = np.array([-1, 1])

class PlayerSelector:
    def __init__(self, 
                 player_names: list[str],
                 kickoff_player_index: int = 2
                ):
        """
        #### Params
        player_names(list[str]): list with all players names.\n
        kickoff_player_index(int): index to pick from sorted list.\n
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
            "Number of agents must be even." + \
            "(i//2 + self.player_count//2) expression" + \
            "breaks with odd player_count."
        
        assert kickoff_player_index == 2 or kickoff_player_index == 3, \
            "First player index must be 2 or 3." + \
            "Index 0 and 1 gets the goalkeeper of team left and right." + \
            "Index 2 and 3 gets the first not goalkeeper player in the player_order_to_play."

        self.player_count = len(player_names)
        
        # Sort player names in the playing rotation order of turns
        self.player_order_to_play = [
            player_names[i//2] if i % 2 == 0 
            else player_names[i//2 + self.player_count//2] 
            for i in range(self.player_count)
        ]

        # _index keeps track of current player
        self._index = kickoff_player_index 

        self._kickoff_player_index = kickoff_player_index
        self._current_player_name = self.player_order_to_play[self._index]
        if kickoff_player_index % 2 == 0:
            self._x_foward_direction = np.array([1, 1])
            self._is_left_team = True
            self._currently_acting_team = TEAM_LEFT_NAME
            self._not_currently_acting_team = TEAM_RIGHT_NAME
            self._next_team_to_play_after_kickoff = TEAM_RIGHT_NAME
        elif kickoff_player_index % 2 != 0:
            self._x_foward_direction = np.array([-1, 1])
            self._is_left_team = False
            self._currently_acting_team = TEAM_RIGHT_NAME
            self._not_currently_acting_team = TEAM_LEFT_NAME
            self._next_team_to_play_after_kickoff = TEAM_LEFT_NAME

        self.selector_logic_callback = self._before_kickoff_logic_callback


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
        print("Trocou time")


    def playing_rotation(self):
        """
        #### Called after kickoff
        The logic to select players change after kickoff.\n
        Change selector logic callback to pick next player considering playing stage.
        """
        self.selector_logic_callback = self._after_kickoff_logic_callback

        if self._next_team_to_play_after_kickoff == TEAM_LEFT_NAME:
            self._index = 0 -1 # Start moving the left team goalkeeper
        else:
            self._index = 1 -1 # Start moving the right team goalkeeper


    def kickoff_rotation(self, team_to_play: str):
        """
        #### Called after goal scored
        The logic to select players change after goal.\n
        Change selector logic callback to pick next player considering kickoff stage.\n
        Needs to recive team_to_play because this class doesn't know who has scored a goal.
        """
        self.selector_logic_callback = self._before_kickoff_logic_callback

        if team_to_play == TEAM_LEFT_NAME:
            self._index = 2 # second player of right team
            self._x_foward_direction = np.array([1, 1])
            self._is_left_team = True
            self._currently_acting_team = TEAM_LEFT_NAME
            self._not_currently_acting_team = TEAM_RIGHT_NAME
            self._next_team_to_play_after_kickoff = TEAM_RIGHT_NAME
        elif team_to_play == TEAM_RIGHT_NAME:
            self._index = 3 # second player of left team
            self._x_foward_direction = np.array([-1, 1])
            self._is_left_team = False
            self._currently_acting_team = TEAM_RIGHT_NAME
            self._not_currently_acting_team = TEAM_LEFT_NAME
            self._next_team_to_play_after_kickoff = TEAM_LEFT_NAME

    
    def _before_kickoff_logic_callback(self):
        # Uncomment the line below if all players of the left team should play before kickoff
        # self_instance._index = (self_instance._index + 2) % self.player_count
        self._current_player_name = self.player_order_to_play[self._index]


    def _after_kickoff_logic_callback(self):
        self._index = (self._index + 1) % self.player_count
        self._current_player_name = self.player_order_to_play[self._index]
        self._toggle_side()
