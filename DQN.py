import os
import gymnasium as gym

import ray
from ray import tune
from ray.air import session, CheckpointConfig

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from ray.rllib.algorithms.dqn import DQNConfig, DQN

from env_setup import env_creator, ENV_NAME
from cnn import CNNModelV2

username = "Ed" # Modifique aqui - Coloque seu nome de usuário

if __name__ == "__main__":

    output_folder = "storage"
    if not os.path.exists("./" + output_folder):
        os.makedirs("./" + output_folder) # Cria a pasta caso não exista

    ray.init(num_cpus=os.cpu_count(), num_gpus=1)

    register_env(
        ENV_NAME, 
        lambda config: PettingZooEnv(env_creator(config))
    )
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        DQNConfig()
        .environment(
            env=ENV_NAME,
            disable_env_checking=True # 'True' devido ao erro "not passing checking"
        )
        .rollouts(
            num_rollout_workers=4
        )
        .training(
            n_step = 10,
            lr = 1e-3,
            gamma = 0.95,
            model={"custom_model": "CNNModelV2"}
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
        
    config.exploration_config.update({ # Decaying epsylon-greedy
        "initial_epsilon": 1.5,
        "final_epsilon": 0.01,
        "epsilon_timesteps": 1_000_000,
    })

    print("Começou run")
    trial_response = tune.run(
        "DQN",
        name="DQN_V3", # Qualquer nome para o experimento
        stop={ 
            # Critério de parada do experimento
            # "episodes_total": 1000,
            # "timesteps_total": 1_000_000, # 1 milhão de passos
            "time_total_s": 300, # 300 segundos
            # "training_iteration": 2,
        },
        checkpoint_config = CheckpointConfig(checkpoint_frequency=10),
        storage_path = f"C:\\Users\\{username}\\Desktop\\gym_soccer_env\\{output_folder}",
        config = config.to_dict(),
        resume = False # resume: [True, False, "LOCAL", "REMOTE", "PROMPT", "AUTO"] para continuar o treino de onde parou
    )

    try:
        print(trial_response.results_df)
        print("=----------------=")
        print(trial_response.results_df["hist_stats/episode_reward"])
        # print(trial_response.results_df["hist_stats/episode_lengths"])
        print("=----------------=")
        # print(trial_response["hist_stats"]) # essa linha da erro
    except Exception as e:
        print(e)

    print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")
