import os
import gymnasium as gym
import env
import random
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType
from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME

""" Esse arquivo serve para debugar manualmente o SoccerEnv.

[1] Definir uma sequência de ações;
[2] Rodar uma sequencia de ações;
[3] Cria uma pasta para salvar as imagens e garante que a pasta esteja vazia;
[4] Salvar na pasta as imagens de cada estado do SoccerEnv;
[5] Inspecionar manualmente cada imagem para verificar se o ambiente executou a sequência corretamente.

=------------------=
A 'action' passada para o 'step' é uma dupla contendo a direção do movimento e a ação.

# posição 0: direção do movimento (de 0 até 8)
# posição 1: ação (de 0 (andar) até 4)

mapeamento da posição 1:
0 -> Right (foward)
1 -> Left (backward)
2 -> Up
3 -> Down
4 -> Right-Up (45° movement)
5 -> Left-Up (45° movement)
6 -> Right-Down (45° movement)
7 -> Left-Down (45° movement)
8 -> No direction

mapeamento da posição 2:
0 -> __move_player
1 -> __steal_ball_action
2 -> __pass_ball
3 -> __kick_ball
4 -> __defend_position
"""

config = {
  "verbose": True
}
actions = []

def first_action_test(team: str):

  config["render_mode"] = 'rgb_array'
  config["observation_format"] = 'dict'
  config["num_agents"] = 11
  config["control_goalkeeper"] = False

  if team == TEAM_RIGHT_NAME:
    config["left_start"] = False
    config["first_player_index"] = 3
  else:
    config["left_start"] = True
    config["first_player_index"] = 2
  actions = [(0,0)]*5 + [(8,0)] # OK
  # actions = [(0,0)]*5 + [(7,0)] # OK
  # actions = [(0,0)]*5 + [(6,0)] # OK
  # actions = [(0,0)]*5 + [(5,0)] # OK
  # actions = [(0,0)]*5 + [(4,0)] # OK
  # actions = [(0,0)]*5 + [(3,0)] # OK
  # actions = [(0,0)]*5 + [(2,0)] # OK
  # actions = [(0,0)]*5 + [(1,0)]*1 # OK
  # actions = [(0,0)]*1 + [(0,0)]*5 # OK
  # actions = [(3,0)]*15 + [(0,0)]*17 + [(0,0)]*2 # OK
  return actions

def right_team_tests():
  config["render_mode"] = 'rgb_array'
  config["observation_format"] = 'dict'
  config["num_agents"] = 8
  config["left_start"] = False
  config["control_goalkeeper"] = False
  config["first_player_index"] = 3
  config["skip_kickoff"] = True

  temp = config["num_agents"] -1
  if not config["control_goalkeeper"]:
    temp = temp -2

  walk_foward_once = [(0,0)]*1 + [(3,0)]*temp
  walk_backward_once = [(1,0)]*1 + [(3,0)]*temp
  walk_up_once = [(2,0)]*1 + [(3,0)]*temp
  walk_down_once = [(3,0)]*1 + [(3,0)]*temp

  # Kick ball tests
  # actions =  walk_foward_once * 10 + [(0,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(0,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(0,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + [(4,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(4,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(4,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + [(6,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(6,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(6,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + [(2,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(2,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(2,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + [(3,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(3,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(3,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + [(8,3)] # OK assert can_kick_to_goal is None
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(8,3)] # OK assert can_kick_to_goal is None
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(8,3)] # OK assert can_kick_to_goal is None

  # Pass ball tests
  # actions = walk_backward_once * 2 + walk_down_once * 1 + [(3, 2)] # [OK] deve passar a bola para o player mais abaixo
  # actions = walk_backward_once * 2 + walk_down_once * 1 + [(2, 2)] # [OK] deve passar a bola para o player mais acima

  # steal ball tests
  actions = walk_foward_once * 3 + [(0,0), (3,0), (3,0), (0,1)] # [OK] player_0 deve roubar a bola de player_5 (ou falhar por azar)

  return actions

def left_team_tests():
  config["render_mode"] = 'rgb_array'
  config["observation_format"] = 'dict'
  config["num_agents"] = 8
  config["left_start"] = True
  config["control_goalkeeper"] = False
  config["first_player_index"] = 2 # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
  config["skip_kickoff"] = True
  
  temp = config["num_agents"] -1
  if not config["control_goalkeeper"]:
    temp = temp -2

  walk_foward_once = [(0,0)]*1 + [(3,0)]*temp
  walk_backward_once = [(1,0)]*1 + [(3,0)]*temp
  walk_up_once = [(2,0)]*1 + [(3,0)]*temp
  walk_down_once = [(3,0)]*1 + [(3,0)]*temp
  kick_to_goal = [(0,3)]*1 + [(3,0)]*temp

  # Walk and kick to goal
  actions = walk_foward_once * 20 + kick_to_goal

  # Kick ball tests
  # actions = walk_foward_once * 10 + [(0,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(0,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(0,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + [(4,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(4,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(4,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + [(6,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(6,3)] # OK assert can_kick_to_goal == True
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(6,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + [(2,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(2,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(2,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + [(3,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(3,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(3,3)] # OK assert can_kick_to_goal == False
  # actions = walk_foward_once * 10 + [(8,3)] # OK assert can_kick_to_goal is None
  # actions = walk_foward_once * 10 + walk_up_once * 4 + [(8,3)] # OK assert can_kick_to_goal is None
  # actions = walk_foward_once * 10 + walk_down_once * 4 + [(8,3)] # OK assert can_kick_to_goal is None

  # Pass ball tests
  # actions = walk_backward_once * 2 + walk_down_once * 1 # 
  
  return actions

# [1] Definir uma sequência de ações;
# actions = first_action_test(TEAM_LEFT_NAME)
# actions = right_team_tests()
actions = left_team_tests()

# [2] Rodar uma sequencia de ações;
env = env.make_raw_env(config)
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
