

# Define constants used in anywhere in the code

WINDOW_SIZE = {'w': 120, 'h': 80} # Pixels in screen = 120 x 80
# WINDOW_SIZE = {'w': 1024, 'h': 768} # Pixels in screen = 1024 x 768
FIELD_SIZE = {'w': 120, 'h': 80}
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
PLAYER_VELOCITY = 2.7
# FIELD_MARGIN = { # Margin between the field rect and the window rect
#     'w': WINDOW_SIZE['w']-FIELD_SIZE['w'] // 2,
#     'h': WINDOW_SIZE['h']-FIELD_SIZE['h'] // 2
# }
POINTS = {
    "center": (WINDOW_SIZE['w'] // 2, WINDOW_SIZE['h'] // 2),
    # "top_left": (FIELD_MARGIN['w'], FIELD_MARGIN['h']),
    # "top_right": (FIELD_MARGIN['w'] + FIELD_SIZE['w'], FIELD_MARGIN['h']),
    # "bottom_left": (FIELD_MARGIN['w'], FIELD_MARGIN['h'] + FIELD_SIZE['h']),
    # "bottom_right": (FIELD_MARGIN['w'] + FIELD_SIZE['w'], FIELD_MARGIN['h'] + FIELD_SIZE['h']),
    # "center_up": (FIELD_MARGIN['w'] + FIELD_SIZE['w'] // 2, FIELD_MARGIN['h']),
    # "center_bottom": (FIELD_MARGIN['w'] + FIELD_SIZE['w'] // 2, FIELD_MARGIN['h'] + FIELD_SIZE['h']),
    # "center_left": (FIELD_MARGIN['w'], FIELD_MARGIN['h'] + FIELD_SIZE['h'] // 2),
    # "center_right": (FIELD_MARGIN['w'] + FIELD_SIZE['w'], FIELD_MARGIN['h'] + FIELD_SIZE['h'] // 2),
    "top_left": (0,0), # Origin
    "top_right": (FIELD_SIZE['w'], 0),
    "bottom_left": (0, FIELD_SIZE['h']),
    "bottom_right": (FIELD_SIZE['w'], FIELD_SIZE['h']),
    "center_up": (FIELD_SIZE['w'] // 2, 0),
    "center_bottom": (FIELD_SIZE['w'] // 2, FIELD_SIZE['h']),
    "center_left": (0, FIELD_SIZE['h'] // 2),
    "center_right": (FIELD_SIZE['w'], FIELD_SIZE['h'] // 2),
}

# Colors
GRAY = (160,160,160)
WHITE = (255,255,255)
GREEN = (0,255,0)
TEAM_LEFT_COLOR = (222, 220, 109)
TEAM_RIGHT_COLOR = (99, 219, 199)
