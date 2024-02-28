import os

# import gym_soccer_env.env
# import gym
import gymnasium

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch

OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig): # python ./run.py +algorithm=dqn env.name='Soccer-v0' env.time_limit=25

    config = {'render_mode': 'rgb_array', 'observation_format': 'dict', 'num_agents' : 6}
    # gym.envs.register(
    #     id="Soccer-v0",
    #     entry_point="gym_soccer_env.env:SoccerEnv",
    #     kwargs=config
    # )
    # gymnasium.envs.registration.register(
    #     id="Soccer-v0",
    #     entry_point="gym_soccer_env.env:SoccerEnv",
    #     kwargs=config
    # )
    # my_env = gym.make("Soccer-v0", **config)
    # my_env = gymnasium.make("Soccer-v0", **config)
    # print(my_env.observation_space)
    # print(my_env.action_space)
    # print(my_env.observation_space())
    # print(my_env.action_space())
    # input(">>> AQUI 1")

    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg.algorithm, _recursive_=False)
    # print("@@@@@@@@@@@@")
    # print(cfg.env)
    # print("@@@@@@@@@@@@")
    env = hydra.utils.call(cfg.env, cfg.seed)

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    # print(env)
    hydra.utils.call(cfg.algorithm, env, logger, _recursive_=False)

    return logger.get_state()

if __name__ == "__main__":

    main()
