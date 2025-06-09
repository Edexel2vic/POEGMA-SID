import os
from tqdm import tqdm
from algorithms import MARLAlgorithm
from solution_concepts import SolutionConcept, MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import numpy as np
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from utils import draw_history
import torch
import torch.nn as nn
import torch.optim as optim
import random
import optuna # --- CAMBIO OPTUNA: Importar la librería ---


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- CAMBIO OPTUNA: Rellenamos la clase DQN para que sea configurable ---
class DQN(nn.Module):
    def __init__(self, num_states, output_size, embedding_dim=64, n_units=128):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, output_size)
        )
    def forward(self, x):
        embedded_state = self.embedding(x)
        return self.network(embedded_state)

class JALGTNN(MARLAlgorithm):
    # Asumo que estás usando nn.Embedding ahora, por eso incluyo embedding_dim
    def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept,
                 gamma=0.95, alpha=0.001, epsilon=0.2, seed=42, n_units=128, embedding_dim=64):
        self.agent_id = agent_id
        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        input_size = self.game.num_states
        output_size = len(self.game.action_space)

        # Usamos la clase DQN con Embedding
        self.q_models = nn.ModuleList([
            DQN(input_size, output_size, embedding_dim=embedding_dim, n_units=n_units) 
            for _ in range(self.game.num_agents)
        ]).to(device)
        
        self.optimizers = [optim.Adam(self.q_models[i].parameters(), lr=self.alpha)
                           for i in range(self.game.num_agents)]
        
        self.loss_fn = nn.MSELoss()
        self.metrics = {"td_error": [], "loss": []}
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        # Versión para nn.Embedding
        return torch.tensor([state], dtype=torch.long).to(device)

    def get_q_values_for_state(self, agent_id, state):
        self.q_models[agent_id].eval()
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values_tensor = self.q_models[agent_id](state_tensor)
        self.q_models[agent_id].train()
        return q_values_tensor

    def value(self, agent_id, state):
        q_values_tensor = self.get_q_values_for_state(agent_id, state)
        # --- FIX: Mover a CPU y APLANAR el array ---
        q_values = q_values_tensor.detach().cpu().numpy()[0]
        
        value = 0
        for idx, joint_action in enumerate(self.game.action_space):
            payoff = q_values[idx]
            joint_probability = np.prod([self.joint_policy[i][state][joint_action[i]]
                                         for i in range(self.game.num_agents)])
            value += payoff * joint_probability
        return value

    def update_policy(self, agent_id, state):
        # --- FIX: Mover a CPU y APLANAR el array ---
        q_values_all_agents = np.array([
            self.get_q_values_for_state(i, state).detach().cpu().numpy()[0]
            for i in range(self.game.num_agents)
        ])
        self.joint_policy[agent_id][state] = self.solution_concept.solution_policy(
            agent_id, state, self.game, q_values_all_agents
        )

    def learn(self, joint_action, rewards, state, next_state):
        joint_action_index = self.game.action_space_index[joint_action]
        
        for agent_id in range(self.game.num_agents):
            agent_reward = rewards[agent_id]
            game_value_next_state = self.value(agent_id, next_state)
            td_target_value = agent_reward + self.gamma * game_value_next_state
            
            state_tensor = self._state_to_tensor(state)
            # El forward pass con nn.Embedding devuelve una forma (1, num_actions)
            predicted_q_values = self.q_models[agent_id](state_tensor) 
            
            target_q_values = predicted_q_values.clone().detach()
            # Accedemos con [0] para modificar la fila correcta del tensor 2D
            target_q_values[0][joint_action_index] = td_target_value 
            
            loss = self.loss_fn(predicted_q_values, target_q_values)
            
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()
            
            self.update_policy(agent_id, state)
            
            # Accedemos con [0] para obtener el valor escalar del tensor
            td_error = td_target_value - predicted_q_values[0][joint_action_index].item()
            self.metrics['td_error'].append(td_error)
            self.metrics['loss'].append(loss.item())

    # ... (resto de métodos sin cambios) ...
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    def solve(self, agent_id, state):
        return self.joint_policy[agent_id][state]

    def select_action(self, state, train=True):
        if train and self.rng.random() < self.epsilon:
            return self.rng.choice(range(self.game.num_actions))
        else:
            probs = self.solve(self.agent_id, state)
            probs = np.clip(probs, 0, 1)
            probs /= np.sum(probs)
            np.random.seed(self.rng.randint(0, 10000))
            return np.random.choice(range(self.game.num_actions), p=probs)
    
    def explain(self, state=0):
        q_values_tensor = self.get_q_values_for_state(self.agent_id, state)
        # --- FIX: Mover a CPU y APLANAR el array ---
        q_values_np = q_values_tensor.detach().cpu().numpy()[0]
        return self.solution_concept.debug(self.agent_id, state, self.game, q_values_np)

# ... (Las funciones obs_to_state, RewardWrapper, create_env no necesitan cambios) ...
def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]
    target = np.max(matrix_target[2]) * 1 + matrix_target[1][0] * 2 + matrix_target[1][2] * 3
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + matrix_obstacles[1][0] * 2 ** 8 + matrix_obstacles[1][2] * 2 ** 7 + matrix_obstacles[2][1] * 2 ** 6
    agents = matrix_agents[0][1] * 2 ** 5 + matrix_agents[1][0] * 2 ** 4 + matrix_agents[1][2] * 2 ** 3 + matrix_agents[2][1] * 2 ** 2
    return int(obstacles + agents + target)

class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, joint_action):
        previous_observations = self.env.unwrapped._obs()
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos

def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config["num_agents"], size=config["size"], density=config["obstacle_density"], seed=seed, max_episode_steps=config["episode_length"], obs_radius=1, on_target="finish", render_mode=None)
    animation_config = AnimationConfig(directory='renders/', static=False, show_agents=True, egocentric_idx=None, save_every_idx_episode=config["save_every"], show_border=True, show_lines=True)
    env = pogema_v0(grid_config)
    #env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)


# --- CAMBIO OPTUNA: Toda la lógica de entrenamiento está ahora en la función `objective` ---
def objective(trial: optuna.Trial) -> float:
    """
    Función que ejecuta un entrenamiento completo y devuelve una métrica a optimizar.
    Optuna llamará a esta función múltiples veces con diferentes hiperparámetros.
    """
    
    # --- CAMBIO OPTUNA: Sugerir hiperparámetros ---
    # Optuna elegirá valores para estos parámetros en cada `trial`
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    n_units = trial.suggest_categorical("n_units", [64, 128, 256])
    episodes_per_epoch = trial.suggest_int("episodes_per_epoch", 10, 50)
    solution_concept_class = trial.suggest_categorical(
            "solution_concept", 
            [
                ParetoSolutionConcept, 
                MinimaxSolutionConcept, 
                NashSolutionConcept, 
                WelfareSolutionConcept
            ]
        )

    exp_config = {
        "num_agents": 2,
        "size": 4,
        "maps": 10,
        "num_states": 16 * 16 * 4,
        "epochs": 200, # Reducimos las épocas para que cada trial sea más rápido
        "episode_length": 16,
        "obstacle_density": 0.1,
        "save_every": None,
        "epsilon_max": 1,
        "epsilon_min": 0.1,
        "renders": "renders/",
        # --- Parámetros controlados por Optuna ---
        "learning_rate": learning_rate,
        "gamma": gamma,
        "n_units": n_units,
        "episodes_per_epoch": episodes_per_epoch,
        "solution_concept": solution_concept_class
    }

    try:
        os.mkdir(exp_config["renders"])
    except:
        pass

    game = GameModel(num_agents=exp_config["num_agents"], num_states=exp_config["num_states"], num_actions=5)
    
    # --- CAMBIO OPTUNA: Pasamos los hiperparámetros sugeridos al algoritmo ---
    algorithms = [JALGTNN(i, game, exp_config["solution_concept"](), 
                        epsilon=exp_config["epsilon_max"],
                        alpha=exp_config["learning_rate"],
                        gamma=exp_config["gamma"],
                        n_units=exp_config["n_units"],
                        seed=i)
                  for i in range(game.num_agents)]

    epsilon_diff = (exp_config["epsilon_max"] - exp_config["epsilon_min"]) / exp_config["episodes_per_epoch"]
    reward_per_epoch = []

    # Bucle de entrenamiento principal (simplificado para el trial)
    for epoch in range(exp_config["epochs"]):
        # Entrenamiento
        for ep in range(exp_config["episodes_per_epoch"]):
            env = create_env(config=exp_config, seed=ep % exp_config["maps"])
            observations, infos = env.reset()
            terminated = truncated = [False, ...]
            states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
            while not all(terminated) and not all(truncated):
                actions = tuple([algorithms[i].select_action(states[i]) for i in range(game.num_agents)])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                
                next_states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                
                # En JALGTNN, cada agente necesita conocer el estado del otro, pero aprende de su propio estado.
                # Para `learn`, usamos el estado y el siguiente estado de cada agente individualmente.
                for i in range(game.num_agents):
                    algorithms[i].learn(actions, rewards, states[i], next_states[i])

                states = next_states

            [algorithms[i].set_epsilon(exp_config["epsilon_max"] - epsilon_diff * ep) for i in range(game.num_agents)]

        # Evaluación
        all_eval_rewards = []
        for ep in range(exp_config["maps"]): # Evaluar en todos los mapas
            env = create_env(config=exp_config, seed=ep)
            observations, infos = env.reset()
            terminated = truncated = [False, ...]
            total_rewards = [0] * exp_config["num_agents"]
            states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
            while not all(terminated) and not all(truncated):
                states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                actions = tuple([algorithms[i].select_action(states[i], train=False)
                                 for i in range(game.num_agents)])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                total_rewards = [total_rewards[i] + rewards[i] for i in range(exp_config["num_agents"])]
            all_eval_rewards.append(sum(total_rewards))
        
        epoch_reward = sum(all_eval_rewards)
        reward_per_epoch.append(epoch_reward)
        
        # --- CAMBIO OPTUNA: Pruning (poda) ---
        # Informa a Optuna del resultado intermedio. Si es un mal trial, Optuna puede detenerlo antes.
        trial.report(epoch_reward, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- CAMBIO OPTUNA: Devolver el valor a optimizar ---
    # Devolvemos la recompensa promedio de las últimas 5 épocas para tener una métrica más estable.
    # Si solo hay una época, devolvemos esa.
    num_epochs_to_average = min(5, len(reward_per_epoch))
    final_metric = np.mean(reward_per_epoch[-num_epochs_to_average:]) if reward_per_epoch else -np.inf
    
    return final_metric


# --- CAMBIO OPTUNA: El bloque principal ahora gestiona el estudio ---
if __name__ == '__main__':
    # 1. Crear el estudio
    # direction="maximize" porque queremos maximizar la recompensa.
    # pruner para detener los ensayos poco prometedores de forma temprana.
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    # 2. Ejecutar la optimización
    # n_trials es el número de combinaciones de hiperparámetros que se probarán.
    # Aumenta este número para una búsqueda más exhaustiva.
    study.optimize(objective, n_trials=50)

    # 3. Imprimir los resultados
    print("\n\n--- OPTIMIZACIÓN COMPLETADA ---")
    print(f"Número de trials finalizados: {len(study.trials)}")
    
    print("\nMejor trial:")
    best_trial = study.best_trial
    print(f"  - Valor (Recompensa): {best_trial.value:.4f}")
    
    print("  - Mejores Hiperparámetros:")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
        
    # Puedes guardar el estudio para reanudarlo más tarde
    # import joblib
    # joblib.dump(study, "marl_study.pkl")
    
    # También puedes visualizar los resultados si tienes plotly instalado
    # pip install plotly
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
    except (ImportError, RuntimeError):
        print("\nPara visualizar los resultados, instala plotly: 'pip install plotly'")