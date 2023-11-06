import gymnasium as gym
import env
import random
from PIL import Image, ImageDraw

env = gym.make("Soccer-v0", render_mode="human", observation_format='dict')

env.reset()

img_start = env.render()

actions = list(range(8))
for __ in range(1):
    for _ in range(22):
        env.step(random.choice(actions))

img_end = env.render()

img_start.show()
img_end.show()

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")

