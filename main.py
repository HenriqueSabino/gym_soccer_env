import gymnasium as gym
import env

num_players = 10
env = gym.make("Soccer-v0", render_mode="human", observation_format='dict', num_agents=num_players)

env.reset()

for __ in range(20):
    for _ in range(num_players):
        sample = env.action_space.sample()
        env.step([sample[0],0])

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")