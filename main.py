import gymnasium as gym
import env
import random

env = gym.make("Soccer-v0", render_mode="human")

env.reset()

actions = list(range(8))
for _ in range(22):
    env.step(random.choice(actions))

print("✅ Successfully run. Remember to look warnings.")

