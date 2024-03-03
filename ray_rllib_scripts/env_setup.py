from env import make_raw_env
from pettingzoo.utils import wrappers as pettingzoo_wrappers
from wrappers.flatten_action_wrapper import FlattenActionWrapper
from wrappers.max_steps_wrapper import MaxStepsWrapper
import gymnasium
import supersuit as ss

def env_creator(args):
    params = {
        'render_mode': 'rgb_array', 
        'observation_format': 'image', 
        'num_agents' : 6,
        'target_score': 1,
        'left_start': True,
        'first_player_index': 2,
        'control_goalkeeper': False,
        'color_option': 2, # Options: [0, 1, 2]
        'skip_kickoff': True,
        'ball_posession_reward': True
    }

    env = make_raw_env(params)
    env = pettingzoo_wrappers.AssertOutOfBoundsWrapper(env)
    env = pettingzoo_wrappers.OrderEnforcingWrapper(env)
    env = FlattenActionWrapper(env)
    env = MaxStepsWrapper(env, max_steps=100)

    # Wrappers ss que processam a observação
    env = ss.color_reduction_v0(env, mode="B") # mode="B" reduz observação RGB para Gray Scale
    env = ss.dtype_v0(env, "float32") # Ambiente tem observação(imagem) com tipo uint8 que deve virar float32
    env = ss.resize_v1(env, x_size=84, y_size=84) # Reduz a imagem do tamanho 120x80 original para 84x84 
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1) # Transforma valores da imagem de [0, 255] para [0, 1]
    env = ss.frame_stack_v1(env, stack_size=3) # Aplica frame stack com 3 frames
    return env
