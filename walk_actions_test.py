import os
import gymnasium as gym
import env
import random
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType

""" Esse arquivo serve para debugar o SoccerEnv.

[1] Definir uma sequência de ações;
[2] Rodar uma sequencia de ações;
[3] Cria uma pasta para salvar as imagens e garante que a pasta esteja vazia;
[4] Salvar na pasta as imagens de cada estado do SoccerEnv;
[5] Inspecionar manualmente cada imagem para verificar se o ambiente executou a sequência corretamente.

=------------------=
A 'action' passada para o 'step' é uma dupla contendo a direção do movimento e a ação.

# posição 0: direção do movimento (de 0 até 8)
# posição 1: ação (de 0 (andar) até 4)
"""

# [1] Definir uma sequência de ações;
# actions = [(0,0)]*5 + [(8,0)] # OK
# actions = [(0,0)]*5 + [(7,0)] # OK
# actions = [(0,0)]*5 + [(6,0)] # OK
# actions = [(0,0)]*5 + [(5,0)] # OK
# actions = [(0,0)]*5 + [(4,0)] # OK
# actions = [(0,0)]*5 + [(3,0)] # OK
# actions = [(0,0)]*5 + [(2,0)] # OK
# actions = [(0,0)]*5 + [(1,0)]*1 # OK
# actions = [(0,0)]*5 + [(0,0)]*1 # OK
actions = [(3,0)]*15 + [(0,0)]*17 + [(0,0)]*2 # OK

# [2] Rodar uma sequencia de ações;
env = gym.make("Soccer-v0", render_mode="human", observation_format='dict', color_option=2)
env.reset()
initial_image = env.render()
images: list[ImageType] = [initial_image]
for action in actions:
  env.step(action)
  images.append(env.render())

# [3] Cria uma pasta para salvar as imagens e garante que a pasta esteja vazia;
output_folder = "./teste_imgs"
if not os.path.exists(output_folder):
  # Cria a pasta caso não exista
  os.makedirs(output_folder)
else:
  # Remove todas as imagens existentes na pasta
  existing_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
  for file in existing_files:
    os.remove(os.path.join(output_folder, file))
  
# [4] Salvar numa pasta a imagem de cada estado do SoccerEnv;
for i, image in enumerate(images):
  image.save(os.path.join(output_folder, f'image_{i+1}.png'))

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")
