import pandas as pd
import matplotlib.pyplot as plt

df_train_1 = pd.read_csv("./train_results/reward_type_1_2player_lr3/statistics.csv")
df_train_2 = pd.read_csv("./train_results/reward_type_2/statistics.csv")

print("=------------=")
print(df_train_1)
print("=------------=")
print(df_train_1.columns)
print("=------------=")

# Suaviza as curvas
def smooth_curve(data, factor=0.6):
    # Applying exponential moving average for smoothing
    return data.ewm(alpha=1 - factor, adjust=False).mean()

# Variáveis
x_1 = df_train_1['steps']
x_2 = df_train_2['steps']
y_team_1 = smooth_curve(df_train_1['team_episode_return'])
y_team_2 = smooth_curve(df_train_2['team_episode_return'])
y_agent1_1 = smooth_curve(df_train_1['left_player_0'])
y_agent2_1 = smooth_curve(df_train_1['left_player_1'])
y_agent1_2 = smooth_curve(df_train_2['left_player_0'])
y_agent2_2 = smooth_curve(df_train_2['left_player_1'])

# Subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot for y_team in df_train_1
axs[0, 0].plot(x_1, y_team_1, label='Team')
axs[0, 0].set_title('Time - Treinamento 1')
axs[0, 0].set_xlabel('Passos')
axs[0, 0].set_ylabel('Retorno da média de 10 episódios')
axs[0, 0].legend()

# Plot for y_team in df_train_2
axs[0, 1].plot(x_2, y_team_2, label='Team')
axs[0, 1].set_title('Time - Treinamento 2')
axs[0, 1].set_xlabel('Passos')
axs[0, 1].set_ylabel('Retorno da média de 10 episódios')
axs[0, 1].legend()

# Plot for y_agent_1 in df_train_1
axs[1, 0].plot(x_1, y_agent1_1, label='Agent 1')
axs[1, 0].plot(x_1, y_agent2_1, label='Agent 2')
axs[1, 0].set_title('Agentes - Treinamento 1')
axs[1, 0].set_xlabel('Passos')
axs[1, 0].set_ylabel('Retorno da média de 10 episódios')
axs[1, 0].legend()

# Plot for y_agent_2 in df_train_2
axs[1, 1].plot(x_2, y_agent1_2, label='Agent 1')
axs[1, 1].plot(x_2, y_agent2_2, label='Agent 2')
axs[1, 1].set_title('Agentes - Treinamento 2')
axs[1, 1].set_xlabel('Passos')
axs[1, 1].set_ylabel('Retorno da média de 10 episódios')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
