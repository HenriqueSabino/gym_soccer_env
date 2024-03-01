# Define constants used in anywhere in the code

# Field constants
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
MID_FIELD_X = FIELD_WIDTH / 2
MID_FIELD_Y = FIELD_HEIGHT / 2
POINTS = {
    "center": (FIELD_WIDTH // 2, FIELD_HEIGHT // 2),
    "top_left": (0,0), # Origin
    "top_right": (FIELD_WIDTH, 0),
    "bottom_left": (0, FIELD_HEIGHT),
    "bottom_right": (FIELD_WIDTH, FIELD_HEIGHT),
    "center_up": (FIELD_WIDTH // 2, 0),
    "center_bottom": (FIELD_WIDTH // 2, FIELD_HEIGHT),
    "center_left": (0, FIELD_HEIGHT // 2),
    "center_right": (FIELD_WIDTH, FIELD_HEIGHT // 2),
}

# Players constants
PLAYER_VELOCITY = 2.7

# Team constants
TEAM_LEFT_NAME = "left_team"
TEAM_RIGHT_NAME = "right_team"

# Ball constants
BALL_VELOCITY = 10.5
BALL_SLOW_CONSTANT = 1.0 # Not used

# Goal constants
GOAL_SIZE = 7.32
TOP_GOAL_Y = FIELD_HEIGHT / 2 - GOAL_SIZE / 2
CENTER_GOAL_Y = FIELD_HEIGHT / 2
BOTTOM_GOAL_Y = FIELD_HEIGHT / 2 + GOAL_SIZE / 2
OUTER_GOAL_HEIGHT = 40 
OUTER_GOAL_WIDTH = 16.5 
INNER_GOAL_HEIGHT = 18.3 
INNER_GOAL_WIDTH = 5.5


# Colors
GRAY             = (160, 160, 160)
WHITE            = (255, 255, 255)
RED              = (255, 0, 0)
GREEN            = (0, 255, 0)
DARK_GREEN       = (15, 84, 21)
BLUE             = (0, 0, 255)
YELLOW           = (249, 230, 54)
BRIGHT_PURPLE    = (197, 40, 240)
TEAM_LEFT_COLOR  = (222, 220, 109)
TEAM_RIGHT_COLOR = (99, 219, 199)
