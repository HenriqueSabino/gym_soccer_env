from copy import deepcopy
import math
import os

import hydra
import torch
from cpprb import ReplayBuffer #, create_before_add_func, create_env_dict
from omegaconf import DictConfig

from MARL_codebase.utils import wrappers
from MARL_codebase.utils.loggers import Logger
from MARL_codebase.utils.video import record_episodes
from MARL_codebase.algorithms.dqn.model import QNetwork 
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
import gymnasium
import numpy as np

from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME

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

        infos.append(env.infos)

    return infos


################################################################################
# Função copiada do arquivo cpprb/util.py da biblioteca cpprb versão 10.2.0
# Essa função tem 2 linhas alteradas para funcionar com ambientes AECEnv

def create_before_add_func(env):
    from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict

    """
    Create function to be used before `ReplayBuffer.add`

    Parameters
    ----------
    env : gym.Env
        Environment for before_func

    Returns
    -------
    before_add : callable
        Function to be used before `ReplayBuffer.add`
    """
    def no_convert(name,v):
        return {f"{name}": v}

    def convert_from_tuple(name,_tuple):
        return {f"{name}{i}": v for i,v in enumerate(_tuple)}

    def convert_from_dict(name,_dict):
        return {f"{name}_{key}":v for key,v in _dict.items()}

    # 2 linhas alteradas
    observation_space = env.observation_space("mock_string")
    action_space = env.action_space("mock_string")

    if isinstance(observation_space, Tuple):
        obs_func = convert_from_tuple
    elif isinstance(observation_space, Dict):
        obs_func = convert_from_dict
    else:
        obs_func = no_convert

    if isinstance(action_space, Tuple):
        act_func = convert_from_tuple
    elif isinstance(action_space, Dict):
        act_func = convert_from_dict
    else:
        act_func = no_convert

    def before_add(obs,act,next_obs,rew,done):
        return {**obs_func("obs",obs),
                **act_func("act",act),
                **obs_func("next_obs",next_obs),
                "rew": rew,
                "done": done}

    return before_add
################################################################################


def main(env: AECEnv, logger: Logger, env_params: dict, cfg: DictConfig, make_env_fn: callable):
    folder_name = cfg.experiment_folder_name

    # replay buffer ############################################################

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
    before_add = create_before_add_func(env)

    ############################################################################

    # Instancia o modelo
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
    
    # Loop principal do treino
    for j in range(cfg.total_steps + 1):
        
        # Testa o modelo
        if j % cfg.eval_interval == 0:
            l = cfg.eval_episodes
            infos = _evaluate(env, model, l, cfg.greedy_epsilon)

            # Prepare data from infos to pass to the logger
            prepared_info_of_each_episode = []
            for info in infos:

                some_agent = env.agents[0]
                return_per_agent = [info[agent]['episode_return'] for agent in env.agents]

                # if env_params['left_start']:
                team_indexes = env.team_to_indexes[TEAM_LEFT_NAME]
                other_team_indexes = env.team_to_indexes[TEAM_RIGHT_NAME]
                team_returns = dict(zip(env.left_agent_names, return_per_agent[team_indexes]))
                other_team_returns = dict(zip(env.right_agent_names, return_per_agent[other_team_indexes]))
                # else:
                #     team_indexes = env.team_to_indexes[TEAM_RIGHT_NAME]
                #     other_team_indexes = env.team_to_indexes[TEAM_LEFT_NAME]
                #     team_returns = dict(zip(env.right_agent_names, return_per_agent[team_indexes]))
                #     other_team_returns = dict(zip(env.left_agent_names, return_per_agent[other_team_indexes]))

                episode_info = {}
                # episode_info["episode_return"]      = info["episode_return"
                episode_info["team_episode_return"] = info[some_agent]["team_episode_return"]
                episode_info["episode_length"]      = info[some_agent]["episode_length"]
                # episode_info["episode_time"]        = i["episode_time"]
                episode_info = {**episode_info, **team_returns, **other_team_returns}
                prepared_info_of_each_episode.append(episode_info)

            # Calculates average of each key. list[dict[(key) str, (value) int | float]]
            final_dict = {}
            for k in prepared_info_of_each_episode[0].keys():
                average = sum(d[k] for d in prepared_info_of_each_episode) / l
                final_dict[k] = [average]
            final_dict['steps'] = j
            final_dict['epsilon'] = eps_sched(j)
            
            # prepared_info_of_each_episode.append(
            #     {'updates': j, 'environment_steps': j, 'epsilon': eps_sched(j)}
            # )

            # Salva métrica/estatísticas em um csv
            logger.log_metrics(final_dict)
            
            # Print média das recompensas do time e de cada agente
            team_episode_return = final_dict["team_episode_return"][0]
            string = f"{l} eval average: team_episode_return [{team_episode_return:.4f}]"
            for agent in env.left_agent_names: # Hardcode iterable
                string += f" | {agent}_return [{final_dict[agent][0]:.4f}]"
            print(string)

        # Modelo escolhe uma ação
        act = model.act(obs, epsilon=eps_sched(j))
        # print("action", act)

        # Aplica ação no ambiente e retorna observação, recompensa, done, truncated, info
        if isinstance(act, int) or isinstance(act, np.int64):
            # print(f"act: {act} | type {type(act)}")
            env.step(act)
        else:
            # print(f"act: {act} | len {len(act)}")
            for a in act:
                env.step(a)
        next_obs, rew, done, truncated, info = env.last()

        # Acrescenta experiência no replay buffer
        rb.add(
            **before_add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)
        )
        
        # Atrasa o treino para coletar experiência no replay buffer
        if j > cfg.training_start:
            batch: dict[str, np.ndarray] = rb.sample(cfg.batch_size)
            # print(batch)
            # print({k: i for i, k in enumerate(batch)})

            batch = [
                torch.from_numpy(v).to(cfg.model.device) for _, v in batch.items()
            ]

            model.update(batch)
        
        # Se terminou reseta
        if done or truncated:
            env.reset()
            obs, reward, terminatd, truncated, info = env.last()
        else:
            obs = next_obs

        # Utiliza o modelo treinado para gravar um vídeo
        if cfg.video_interval and j % cfg.video_interval == 0:
            record_episodes(
                # deepcopy(env), # Deep copy não funciona com ambientes AECEnv
                make_env_fn, # Essa função que cria o env do zero substitui deepcopy
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./train_results/{folder_name}/videos/step-{j}.mp4",
                env_params
            )
        
    torch.save(model.state_dict(), f"./train_results/{folder_name}/{cfg.model_name}_trained_model.tch")
