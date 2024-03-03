from copy import deepcopy
import math

import hydra
import torch
from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from omegaconf import DictConfig

from MARL_codebase.utils import wrappers
from MARL_codebase.utils.loggers import Logger
from MARL_codebase.utils.video import record_episodes
from MARL_codebase.algorithms.dqn.model import QNetwork 
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
import gymnasium
import numpy as np

def _epsilon_schedule(decay_style, eps_start, eps_end, eps_decay, total_steps):
    """
    Exponential decay schedule for exploration epsilon.
    :param decay_style: Style of epsilon schedule. One of "linear"/ "lin" or "exponential"/ "exp".
    :param eps_start: Starting epsilon value.
    :param eps_end: Ending epsilon value.
    :param eps_decay: Decay rate.
    :param total_steps: Total number of steps to take.
    :return: Epsilon schedule function mapping step number to epsilon value.
    """
    assert decay_style in ["linear", "lin", "exponential", "exp"], "decay_style must be one of 'linear' or 'exponential'"
    assert 0 <= eps_start <= 1 and 0 <= eps_end <= 1, "eps must be in [0, 1]"
    assert eps_start >= eps_end, "eps_start must be >= eps_end"
    assert total_steps > 0, "total_steps must be > 0"
    assert eps_decay > 0, "eps_decay must be > 0"
    
    if decay_style in ["linear", "lin"]:
        def _thunk(steps_done):
            return eps_end + (eps_start - eps_end) * (1 - steps_done / total_steps)
    elif decay_style in ["exponential", "exp"]:
        eps_decay = (eps_start - eps_end) / total_steps * eps_decay
        def _thunk(steps_done):
            return eps_end + (eps_start - eps_end) * math.exp(-eps_decay * steps_done)
    else:
        raise ValueError("decay_style must be one of 'linear' or 'exponential'")
    return _thunk


def _evaluate(env, model, eval_episodes, greedy_epsilon, verbose=False):
    infos = []
    for j in range(eval_episodes):
        done = False
        env.reset()
        obs, reward, done, truncated, info = env.last()
        while not done and not truncated:
            # input(">>>")
            with torch.no_grad():
                act = model.act(obs, greedy_epsilon)
                # print("ACT from dqn: ", act)
            if isinstance(act, int) or isinstance(act, np.int64):
                if verbose:
                    print(f"act: {act} | type {type(act)}")
                env.step(act)
            else:
                if verbose:
                    print(f"act: {act} | len {len(act)}")
                for a in act:
                    env.step(a)
            obs, _, done, truncated, info = env.last()

        infos.append(info)

    return infos


def main(env: AECEnv, logger: Logger, **cfg):

    cfg = DictConfig(cfg)

    # replay buffer:
    # env_dict = create_env_dict(env, int_type=np.uint8, float_type=np.float32)
    observation_space = env.observation_space("mock_string") # Space.Box
    action_space = env.action_space("mock_string") # Space.Discrete
    env_dict = {
        "rew" : {"shape": 2, "dtype": np.float32},
        "done": {"shape": 1, "dtype": np.float32},
        "obs" : {"shape": observation_space.shape, "dtype": np.float32},
        "next_obs" : {"shape": observation_space.shape, "dtype": np.float32},
        "act" : {"shape": 2, "dtype": np.float32} # np.uint8
    }

    # print(cfg.buffer_size)
    # print(env_dict)
    rb = ReplayBuffer(cfg.buffer_size, env_dict)
    # print(rb)
    # input(">>> ver rb")
    before_add = create_before_add_func(env)
    model = QNetwork(
        env.half_number_agents,
        observation_space,
        action_space,
        cfg,
        cfg.model.layers,
        cfg.model.parameter_sharing,
        cfg.model.use_orthogonal_init,
        cfg.model.device
    )

    # Logging
    logger.watch(model)

    # epsilon
    eps_sched = _epsilon_schedule(cfg.eps_decay_style, cfg.eps_start, cfg.eps_end, cfg.eps_decay, cfg.total_steps)

    # training loop:
    env.reset()
    obs, reward, terminatd, truncated, reset_infos = env.last()

    # print("@@@@@@@@@@@@@@@")
    # print(cfg)
    # print("@@@@@@@@@@@@@@@")
    # print(env)
    # print(obs)
    # print(obs.shape)
    # print("@@@@@@@@@@@@@@@")

    # input(">>> dqn.train.main 1")
    

    for j in range(cfg.total_steps + 1):
        
        # input(">>> dqn.train.main 2")
        if j % cfg.eval_interval == 0:
            infos = _evaluate(env, model, cfg.eval_episodes, cfg.greedy_epsilon)
            # infos = _evaluate(env, model, 1, cfg.greedy_epsilon)

            # input(">>> dqn.train.main 3")
            # Prepare data from infos to pass to the logger
            prepared_info_of_each_episode = []
            for i in infos:
                # print(i)
                episode_info = {}
                # selected_player_name = env.agent_selection
                # episode_info["episode_return"]      = i[selected_player_name]["episode_return"]
                # episode_info["team_episode_return"] = i[selected_player_name]["team_episode_return"]
                # episode_info["episode_length"]      = i[selected_player_name]["episode_length"]
                # episode_info["episode_time"]        = i[selected_player_name]["episode_time"]
                episode_info["episode_return"]      = i["episode_return"]
                episode_info["team_episode_return"] = i["team_episode_return"]
                episode_info["episode_length"]      = i["episode_length"]
                episode_info["episode_time"]        = i["episode_time"]

                prepared_info_of_each_episode.append(episode_info)
            prepared_info_of_each_episode.append(
                {'updates': j, 'environment_steps': j, 'epsilon': eps_sched(j)}
            )
            # print("infos")
            # print(prepared_info_of_each_episode)
            # print("=------=")
            # input(">>> dqn.train.main 4")
            logger.log_metrics(prepared_info_of_each_episode)
            episode_return = episode_info["episode_return"]
            print(f"eval episode return: {episode_return}")

        
        # input(">>> dqn.train.main 5")
        act = model.act(obs, epsilon=eps_sched(j))
        # print("action", act)
        # input(">>> dqn.train.main 6")
        if isinstance(act, int) or isinstance(act, np.int64):
            # print(f"act: {act} | type {type(act)}")
            env.step(act)
        else:
            # print(f"act: {act} | len {len(act)}")
            for a in act:
                env.step(a)
        next_obs, rew, done, truncated, info = env.last()

        # input(">>> dqn.train.main 7")

        # if (cfg.use_proper_termination and (done or truncated) and info.get("TimeLimit.truncated", False)):
        #     del info["TimeLimit.truncated"]
        #     proper_done = False
        # if cfg.use_proper_termination and truncated:
        #     proper_done = False
        # elif cfg.use_proper_termination == "ignore":
        #     proper_done = False
        # else:
        #     proper_done = done
        # print(obs)
        # print(act)
        # print(next_obs)
        # print(rew)
        # print(proper_done)
        rb.add(
            **before_add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)
        )
        # input(">>> dqn.train.main 8")
        

        if j > cfg.training_start:
            batch: dict[str, np.ndarray] = rb.sample(cfg.batch_size)
            # print(type(batch))
            # print(batch)
            # input(">>> anotar typehint do batch")

            # Usa isso pra setar o dict labels_to_index no model.py
            # print({k: i for i, k in enumerate(batch)})

            batch = [
                torch.from_numpy(v).to(cfg.model.device) for _, v in batch.items()
            ]

            model.update(batch)
        # input(">>> dqn.train.main 9")
        
        if done or truncated:
            env.reset()
            obs, reward, terminatd, truncated, info = env.last()
        else:
            obs = next_obs
        # input(">>> dqn.train.main 10")

        if cfg.video_interval and j % cfg.video_interval == 0:
            record_episodes(
                deepcopy(env),
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./videos/step-{j}.mp4",
            )
            
        # input(">>> dqn.train.main 11")
        
    torch.save(model.state_dict(), "./trained_models/latest_trained_model.tch")


if __name__ == "__main__":
    main()
