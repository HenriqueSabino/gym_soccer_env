import gymnasium as gym
import env
import random
from PIL import Image, ImageDraw

env = gym.make("Soccer-v0", render_mode="human")

env.reset()

env.render().show()

actions = list(range(8))
for _ in range(22):
    env.step(random.choice(actions))

env.render().show()

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")

