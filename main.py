import gymnasium as gym
import env

env = gym.make("Soccer-v0", render_mode="human", observation_format='dict')

env.reset()

for __ in range(20):
    for _ in range(22):
        sample = env.action_space.sample()
        env.step([sample[0],0])

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")