import env # Esse import registra o SoccerEnv no gymnasium
import os

algorithms = ["ac", "dqn", "vdn", "qmix"]
command = f"python run.py +algorithm=dqn env.name='Soccer-v0' env.time_limit=25"
os.chdir("./MARL-codebase/fastmarl")
os.system(command)

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")
