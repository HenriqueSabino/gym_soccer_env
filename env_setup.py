import env
import gymnasium as gym
import supersuit as ss

# Environment setup
ENV_NAME = "Soccer-v0"

def env_creator(args):
    env = gym.make(
        ENV_NAME, 
        render_mode = "rgb_array", 
        observation_format = 'image', 
        num_agents = 4
    )

    # Wrappers que processam a observação
    env = ss.color_reduction_v0(env, mode="B") # mode="B" reduz observação RGB para Gray Scale
    env = ss.dtype_v0(env, "float32") # Ambiente tem observação(imagem) com tipo uint8 que deve virar float32
    env = ss.resize_v1(env, x_size=84, y_size=84) # Reduz a imagem do tamanho 512x512 original para 84x84 
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1) # Transforma valores da imagem de [0, 255] para [0, 1]
    env = ss.frame_stack_v1(env, stack_size=3) # Aplica frame stack com 3 frames
    return env
