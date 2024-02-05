import gymnasium as gym
import env

env = gym.make("Soccer-v0", render_mode="human", observation_format='dict')

env.reset()

img_start = env.render()

for __ in range(20):
    for _ in range(22):
        env.step(env.action_space.sample())

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")