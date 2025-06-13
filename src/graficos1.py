import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar los datos
df = pd.read_csv('src/out/jalgt_minimaxsolutionconcept_optuna_results_detailed.csv')

# Análisis básico

# Configurar el estilo de los gráficos
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Evolución del valor objetivo durante la optimización
plt.subplot(1, 2, 1)
plt.plot(df['number'], df['value'], 'b-', label='Valor objetivo')
plt.xlabel('Número de trial')
plt.ylabel('Valor objetivo')
plt.title('Evolución del valor objetivo')
plt.legend()

# Distribución del valor objetivo
plt.subplot(1, 2, 2)
sns.histplot(df['value'], bins=20, kde=True)
plt.xlabel('Valor objetivo')
plt.ylabel('Frecuencia')
plt.title('Distribución del valor objetivo')

plt.tight_layout()
plt.show()

# Análisis de hiperparámetros
params = ['episode_length', 'epsilon_max', 'epsilon_min', 'gamma', 'learning_rate', 'num_episodes']

plt.figure(figsize=(15, 10))
for i, param in enumerate(params, 1):
    plt.subplot(2, 3, i)
    plt.scatter(df[f'params_{param}'], df['value'], alpha=0.5)
    plt.xlabel(param)
    plt.ylabel('Valor objetivo')
    plt.title(f'Impacto de {param}')

plt.tight_layout()
plt.show()

# Correlación entre parámetros y valor objetivo
corr_matrix = df[[f'params_{p}' for p in params + ['value']]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de correlación')
plt.show()

# Análisis de las recompensas
reward_cols = ['user_attrs_collective_reward_mean', 
              'user_attrs_individual_reward_agent0_mean',
              'user_attrs_total_individual_reward_agent0_mean']

plt.figure(figsize=(15, 5))
for i, col in enumerate(reward_cols, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=df['value'], y=df[col])
    plt.xlabel('Valor objetivo')
    plt.ylabel(col.replace('user_attrs_', '').replace('_', ' ').title())
    plt.title(f'Relación con {col.split("_")[-2]}')

plt.tight_layout()
plt.show()

# Tiempos de entrenamiento
time_cols = ['user_attrs_training_time_per_episode_mean', 
            'user_attrs_training_time_total_mean']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['duration'], y=df['value'])
plt.xlabel('Duración del trial (segundos)')
plt.ylabel('Valor objetivo')

plt.subplot(1, 2, 2)
sns.scatterplot(x=df[time_cols[1]], y=df['value'])
plt.xlabel('Tiempo total de entrenamiento')
plt.ylabel('Valor objetivo')

plt.tight_layout()
plt.show()

# Análisis de las ejecuciones paralelas
runs = ['run_0', 'run_1', 'run_2', 'run_3']
collective_rewards = [df[f'user_attrs_{r}_collective_reward'] for r in runs]

plt.figure(figsize=(10, 6))
for i, rewards in enumerate(collective_rewards, 1):
    sns.kdeplot(rewards, label=f'Ejecución {i-1}')
plt.xlabel('Recompensa colectiva')
plt.ylabel('Densidad')
plt.title('Distribución de recompensas por ejecución paralela')
plt.legend()
plt.show()

# Mejor trial
best_trial = df.loc[df['value'].idxmax()]
print("\nDetalles del mejor trial:")
print(f"Número: {best_trial['number']}")
print(f"Valor: {best_trial['value']}")
print(f"Duración: {best_trial['duration']} segundos")
print("Parámetros:")
for param in params:
    print(f"  {param}: {best_trial[f'params_{param}']}")
print(f"Recompensa colectiva media: {best_trial['user_attrs_collective_reward_mean']}")
print(f"Recompensa individual media (agente 0): {best_trial['user_attrs_individual_reward_agent0_mean']}")
