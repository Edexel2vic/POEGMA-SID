import os
import csv
from tqdm import tqdm
from algorithms import MARLAlgorithm
from solution_concepts import SolutionConcept, MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import numpy as np
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
# from utils import draw_history # Comentado si no se usa
import torch
import torch.nn as nn
import torch.optim as optim
import random
import optuna
import collections
from typing import List, Tuple
from collections import deque, namedtuple
import random

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")



class DQN(nn.Module):
    """Red Neuronal Profunda para aproximar la función Q."""
    def __init__(self, num_states: int, output_size: int, embedding_dim: int = 64, n_units: int = 128):
        super(DQN, self).__init__()
        # Capa de embedding para manejar estados discretos de gran cardinalidad
        self.embedding = nn.Embedding(num_states, embedding_dim)
        # Red neuronal feed-forward
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded_state = self.embedding(x)
        return self.network(embedded_state)


class JALGTNN(MARLAlgorithm):
    """
    Joint Action Learners with Game-Theoretic Neural Networks.
    Un algoritmo MARL que utiliza redes neuronales para aprender los valores Q
    y conceptos de teoría de juegos para derivar políticas a partir de esos valores.
    """
    def __init__(self, game: GameModel, solution_concept: SolutionConcept,
                 gamma: float = 0.95, alpha: float = 0.001, epsilon: float = 0.2, 
                 seed: int = 42, n_units: int = 128, embedding_dim: int = 64):

        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        input_size = self.game.num_states
        # El output size es el número de acciones conjuntas
        output_size = len(self.game.action_space)

        self.q_models = nn.ModuleList([
            DQN(input_size, output_size, embedding_dim=embedding_dim, n_units=n_units)
            for _ in range(self.game.num_agents)
        ]).to(device)

        self.optimizers = [optim.Adam(self.q_models[i].parameters(), lr=self.alpha)
                           for i in range(self.game.num_agents)]

        self.loss_fn = nn.MSELoss()
        self.metrics = [{"td_error": [], "loss": []} for i in range(self.game.num_agents)]

        # Almacena las políticas individuales para cada agente en cada estado.
        # Shape: (num_agentes, num_estados, num_acciones_individuales)
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """Convierte un estado entero a un tensor de PyTorch."""
        return torch.tensor([state], dtype=torch.long).to(device)

    def get_q_values_for_state(self, agent_id: int, state: int) -> torch.Tensor:
        """Obtiene los valores Q para un agente y estado dados."""
        self.q_models[agent_id].eval()
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values_tensor = self.q_models[agent_id](state_tensor)
        self.q_models[agent_id].train()
        return q_values_tensor

    def value(self, agent_id: int, state: int) -> float:
        """Calcula el valor esperado de un estado para un agente, basado en la política conjunta actual."""
        q_values_tensor = self.get_q_values_for_state(agent_id, state)
        q_values = q_values_tensor.detach().cpu().numpy()[0]

        value = 0.0
        # NOTA: Esta iteración puede ser lenta si el espacio de acciones conjuntas es grande.
        for idx, joint_action in enumerate(self.game.action_space):
            payoff = q_values[idx]
            joint_probability = np.prod([self.joint_policy[i, state, joint_action[i]]
                                         for i in range(self.game.num_agents)])
            value += payoff * joint_probability
        return value

    def update_policy(self, agent_id: int, state: int):
        """Actualiza la política para un agente en un estado dado usando el concepto de solución."""
        q_values_all_agents = np.array([
            self.get_q_values_for_state(i, state).detach().cpu().numpy()[0]
            for i in range(self.game.num_agents)
        ])
        # La política se calcula y se actualiza para el agente específico
        self.joint_policy[agent_id, state] = self.solution_concept.solution_policy(
            agent_id, state, self.game, q_values_all_agents
        )

    def learn(self, joint_action: Tuple[int, ...], rewards: List[float], 
              states: List[int], next_states: List[int], terminated: Tuple[bool, ...]):
        
        joint_action_index = self.game.action_space_index[joint_action]

        for agent_id in range(self.game.num_agents):
            agent_reward = rewards[agent_id]

            if terminated[agent_id]:
                game_value_next_state = 0.0
            else:
                game_value_next_state = self.value(agent_id, next_states[agent_id])
            
            td_target_value = agent_reward + self.gamma * game_value_next_state

            state_tensor = self._state_to_tensor(states[agent_id])
            predicted_q_values = self.q_models[agent_id](state_tensor)

            target_q_values = predicted_q_values.clone().detach()
            target_q_values[0, joint_action_index] = td_target_value

            loss = self.loss_fn(predicted_q_values, target_q_values)

            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()

            self.update_policy(agent_id, states[agent_id])

            td_error = td_target_value - predicted_q_values[0, joint_action_index].item()
            self.metrics[agent_id]['td_error'].append(td_error)
            self.metrics[agent_id]['loss'].append(loss.item())

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon
        
    def select_action(self, agent_id: int, state: int, train: bool = True) -> int:
        """
        Selecciona una acción para un agente específico en un estado dado.
        Usa una estrategia epsilon-greedy sobre la política calculada.
        """
        # **CORRECCIÓN CRÍTICA**: Se ha añadido agent_id como argumento.
        if train and self.rng.random() < self.epsilon:
            # Exploración: elige una acción individual aleatoria
            return self.rng.choice(range(self.game.num_actions))
        else:
            # Explotación: usa la política del agente
            # **CORRECCIÓN**: Accede a la política del agente correcto.
            probs = self.joint_policy[agent_id, state]
            # Asegurarse de que las probabilidades son válidas
            probs = np.clip(probs, 0, 1)
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs /= probs_sum
            else:
                # Si la suma es 0, recurrir a una política uniforme
                probs = np.ones(self.game.num_actions) / self.game.num_actions

            return np.random.choice(range(self.game.num_actions), p=probs)

    def explain(self, agent_id: int, state: int = 0):
        """Explica la decisión de un agente en un estado."""
        # **CORRECCIÓN**: Se ha añadido agent_id como argumento.
        q_values_all_agents = np.array([
            self.get_q_values_for_state(i, state).detach().cpu().numpy()[0]
            for i in range(self.game.num_agents)
        ])
        return self.solution_concept.debug(agent_id, state, self.game, q_values_all_agents)


# --- Las funciones auxiliares no necesitan cambios ---
def obs_to_state(obs):
    """Convierte una observación de Pogema en un estado entero único."""
    # Esta función parece estar codificando la información de la cuadrícula de 3x3
    # en un solo entero usando potencias de 2 (como una máscara de bits).
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]
    target = np.max(matrix_target[2]) * 1 + matrix_target[1][0] * 2 + matrix_target[1][2] * 3
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + matrix_obstacles[1][0] * 2 ** 8 + matrix_obstacles[1][2] * 2 ** 7 + matrix_obstacles[2][1] * 2 ** 6
    agents = matrix_agents[0][1] * 2 ** 5 + matrix_agents[1][0] * 2 ** 4 + matrix_agents[1][2] * 2 ** 3 + matrix_agents[2][1] * 2 ** 2
    return int(obstacles + agents + target)

class RewardWrapper(Wrapper):
    """Añade una pequeña penalización por paso para incentivar la rapidez."""
    def __init__(self, env):
        super().__init__(env)
    def step(self, joint_action):
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:
                    rewards[i] = -0.01  # Penalización por paso
        return observations, rewards, terminated, truncated, infos

def create_env(config, seed=42):
    """Crea una instancia del entorno Pogema."""
    grid_config = GridConfig(
        num_agents=config["num_agents"], size=config["size"], density=config["obstacle_density"], 
        seed=seed, max_episode_steps=config["episode_length"], obs_radius=1, 
        on_target="finish", render_mode=None
    )
    env = pogema_v0(grid_config)
    return RewardWrapper(env)


def objective(trial: optuna.Trial) -> float:
    """Función objetivo para la optimización de hiperparámetros con Optuna."""
    # Sugerir hiperparámetros
    learning_rate = trial.suggest_float("learning_rate", 0.000001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    n_units = trial.suggest_categorical("n_units", [64, 128, 256])
    epsilon_min = trial.suggest_float("epsilon_min", 0.001, 1.0)
    episode_length = trial.suggest_int("episode_length", 5, 70)


    exp_config = {
        "num_agents": 2, "size": 4, "maps": 10, "num_states": 4*16*16, # 4096 estados posibles
        "epochs": 10, "episode_length": episode_length, "obstacle_density": 0.1,
        "save_every": None, "epsilon_max": 1.0, "epsilon_min": epsilon_min,
        "renders": "renders/",
        "learning_rate": learning_rate, "gamma": gamma, "n_units": n_units,
        "solution_concept": ParetoSolutionConcept
    }
    
            
    game = GameModel(num_agents=exp_config["num_agents"], num_states=exp_config["num_states"], num_actions=5)


    algorithms = JALGTNN(game, exp_config["solution_concept"](),
                         epsilon=exp_config["epsilon_max"],
                         alpha=exp_config["learning_rate"],
                         gamma=exp_config["gamma"],
                         n_units=exp_config["n_units"],
                         seed=3)

    total_epsiodes = exp_config["episode_length"] * exp_config["epochs"]
    epsilon_diff = (exp_config["epsilon_max"] - exp_config["epsilon_min"]) / total_epsiodes
    current_epsilon = exp_config["epsilon_max"]
    reward_per_epoch = []

    for epoch in range(exp_config["epochs"]):
        # Entrenamiento
        for ep in range(exp_config["episode_length"]):
            env = create_env(config=exp_config, seed=ep % exp_config["maps"])
            observations, infos = env.reset()
            terminated = truncated = [False] * exp_config["num_agents"]
            states = [obs_to_state(obs) for obs in observations]
            
            while not all(terminated) and not all(truncated):
                # **CORRECCIÓN**: Pasar agent_id y el estado de ese agente a select_action
                actions = tuple([algorithms.select_action(i, states[i], train=True) for i in range(game.num_agents)])
                
                next_observations, rewards, terminated, truncated, infos = env.step(actions)
                next_states = [obs_to_state(obs) for obs in next_observations]
                
                algorithms.learn(actions, rewards, states, next_states, tuple(terminated))
                
                states = next_states

            current_epsilon = max(exp_config["epsilon_min"], current_epsilon - epsilon_diff)
            algorithms.set_epsilon(current_epsilon)

        # Evaluación
    all_eval_rewards = []
    collective_reward = [] # Métrica de éxito (cuántos agentes llegan a la meta)
    for ep in range(exp_config["maps"]):
        env = create_env(config=exp_config, seed=ep)
        observations, infos = env.reset()
        terminated = truncated = [False] * exp_config["num_agents"]
        total_rewards = [0.0] * exp_config["num_agents"]
            
        while not all(terminated) and not all(truncated):
            states = [obs_to_state(obs) for obs in observations]
                
            # **CORRECCIÓN**: Pasar agent_id y el estado de ese agente a select_action
            actions = tuple([algorithms.select_action(i, states[i], train=False) for i in range(game.num_agents)])
                
            observations, rewards, terminated, truncated, infos = env.step(actions)
            total_rewards = [total_rewards[i] + rewards[i] for i in range(exp_config["num_agents"])]
                
        all_eval_rewards.append(sum(total_rewards))
        collective_reward.append(sum(terminated)) # Suma de booleanos (True=1, False=0)

    epoch_reward = np.mean(all_eval_rewards)
    reward_per_epoch.append(epoch_reward)
        

    final_metric = sum(all_eval_rewards)
    collective_reward_mean = sum(collective_reward)
    trial.set_user_attr("collective_reward_mean", collective_reward_mean)
    
    return final_metric


def save_results_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """Callback para guardar los resultados de cada trial de Optuna en un archivo CSV."""
    CSV_FILE = "pareto_nn.csv"

    # Encabezado del CSV (consistente para cada escritura)
    params_keys = list(trial.params.keys())
    header = [
        "trial_number",          # Número del trial
        "individual_reward",     # Métrica objetivo (recompensa promedio individual)
        "collective_reward",     # Métrica extra (promedio de agentes que llegaron a la meta)
        "duration_seconds",      # Duración del trial
        "state"                  # Estado final (COMPLETE, PRUNED, FAIL)
    ] + params_keys
    
    # Preparar la fila de datos para el trial actual
    duration = trial.duration.total_seconds() if trial.duration else 0
    state = trial.state.name
    value = trial.value if trial.value is not None else "N/A"
    collective_reward_mean = trial.user_attrs.get("collective_reward_mean", "N/A")
    params_values = [trial.params.get(key) for key in params_keys]
    
    data_row = [trial.number, value, collective_reward_mean, duration, state] + params_values

    # Escribir en el archivo CSV
    file_exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)  # Escribir encabezado si el archivo es nuevo
            writer.writerow(data_row)
        print(f"Trial {trial.number} finalizado. Resultados guardados en '{CSV_FILE}'.")
    except IOError as e:
        print(f"Error al escribir en el archivo CSV: {e}")


if __name__ == '__main__':
    # 1. Crear el estudio de Optuna
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    # 2. Ejecutar la optimización con el callback
    study.optimize(
        objective,
        n_trials=100,
        callbacks=[save_results_callback] # Añadir el callback aquí
    )

    # 3. Imprimir los resultados finales
    print("\n\n--- OPTIMIZACIÓN COMPLETADA ---")
    print(f"Número de trials finalizados: {len(study.trials)}")

    print("\nMejor trial:")
    best_trial = study.best_trial
    print(f"  - Valor (Recompensa Promedio): {best_trial.value:.4f}")
    
    print("  - Mejores Hiperparámetros:")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")

    print(f"\nLos resultados de todos los trials se han guardado en '{CSV_FILE}'.")

    # 4. Visualización de resultados (opcional)
    try:
        if optuna.visualization.is_available():
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()

            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.show()
    except (ImportError, RuntimeError):
        print("\nPara visualizar los resultados, instala plotly: 'pip install plotly>=4.0.0'")