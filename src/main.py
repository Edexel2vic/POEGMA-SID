# main.py

import os
import time
from tqdm import tqdm
from algorithms import JALGT, IQLAgent
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import numpy as np
from gymnasium import Wrapper
import optuna 
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from utils import draw_history


def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]
    target = np.max(matrix_target[2]) * 1 + \
             matrix_target[1][0] * 2 + matrix_target[1][2] * 3
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + \
                matrix_obstacles[1][0] * 2 ** 8 + \
                matrix_obstacles[1][2] * 2 ** 7 + \
                matrix_obstacles[2][1] * 2 ** 6
    agents = matrix_agents[0][1] * 2 ** 5 + \
             matrix_agents[1][0] * 2 ** 4 + \
             matrix_agents[1][2] * 2 ** 3 + \
             matrix_agents[2][1] * 2 ** 2
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
    grid_config = GridConfig(num_agents=config["num_agents"],
                             size=config["size"],
                             density=config["obstacle_density"],
                             seed=seed,
                             max_episode_steps=config["episode_length"],
                             obs_radius=1,
                             on_target="finish",
                             render_mode=None)
    # Disable animations during hyperparameter search for speed
    if config.get("save_every") is not None:
        animation_config = AnimationConfig(directory='renders/',
                                           static=False,
                                           show_agents=True,
                                           egocentric_idx=None,
                                           save_every_idx_episode=config["save_every"],
                                           show_border=True,
                                           show_lines=True)
        env = pogema_v0(grid_config)
        env = AnimationMonitor(env, animation_config=animation_config)
    else:
        env = pogema_v0(grid_config)
        
    return RewardWrapper(env)


# We wrap the main logic in a function that takes the config and returns the score
def run_experiment(config, pbar=None):
    """
    Runs a single experiment with a given configuration.
    Returns the final collective evaluation reward.
    """
    game = GameModel(num_agents=config["num_agents"], num_states=config["num_states"],
                     num_actions=5)
    
    # IMPORTANT: Ensure the algorithm uses the hyperparameters from the 
    if config["algorithm"] == "JALGT":
        algorithms = [JALGT(agent_id=i,
                           game=game,
                           solution_concept=config["solution_concept"],
                           gamma=config["gamma"],
                           alpha=config["learning_rate"],
                           epsilon=config["epsilon_max"],
                           seed=1) 
                    for i in range(game.num_agents)]
        
    elif config["algorithm"] == "IQL":
        algorithms = [IQLAgent(agent_id=i, 
                           num_local_states=config["num_states"], 
                           num_individual_actions=5, 
                           epsilon_start=config["epsilon_max"], 
                           epsilon_end=config["epsilon_min"],
                           gamma=config["gamma"], # Use gamma from config
                           alpha=config["learning_rate"], 
                           seed=i)
                  for i in range(game.num_agents)]

    epsilon_diff = (config["epsilon_max"] - config["epsilon_min"]) / config["episodes_per_epoch"]
    
    # We remove metric lists from the main function as Optuna only needs the final score
    # reward_per_epoch = []
    # td_error_per_epoch = []

    num_epochs = config["epochs"]
    if pbar:
        iterator = range(num_epochs)
    else:
        # Using tqdm if no external progress bar is provided
        iterator = tqdm(range(num_epochs), desc="Running Experiment")

    for epoch in iterator:
        # Training
        for ep in range(config["episodes_per_epoch"]):
            if pbar: pbar.set_postfix({'modo': 'entrenamiento', 'episodio': ep})
            
            env = create_env(config=config, seed=ep % config["maps"])
            observations, infos = env.reset()
            terminated = truncated = [False, ...]
            states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
            
            while not all(terminated) and not all(truncated):
                actions = tuple([algorithms[i].select_action(states[i]) for i in range(game.num_agents)])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                experience_batch = {
                        'joint_action': actions, 'rewards': rewards,
                        'current_global_state': states[0], 'next_global_state': obs_to_state(observations[0]),
                        'current_local_states': states, 'next_local_states': [obs_to_state(obs) for obs in observations],
                        'dones': terminated
                    }
                [algorithms[i].learn(experience_batch) for i in range(game.num_agents)]
                states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

            [algorithms[i].set_epsilon(config["epsilon_max"] - epsilon_diff * ep) for i in range(game.num_agents)]

    # Evaluation (after all training epochs are done)
    evaluation_episodes = config["maps"]
    all_eval_rewards = []
    for ep in range(evaluation_episodes):
        if pbar: pbar.set_postfix({'modo': 'evaluación...', 'episodio': ep})
        
        env = create_env(config=config, seed=ep)
        observations, infos = env.reset()
        terminated = truncated = [False, ...]
        total_rewards = [0] * config["num_agents"]
        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
        
        while not all(terminated) and not all(truncated):
            states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
            actions = tuple([algorithms[i].select_action(states[i], train=False)
                             for i in range(game.num_agents)])
            observations, rewards, terminated, truncated, infos = env.step(actions)
            total_rewards = [total_rewards[i] + rewards[i] for i in range(config["num_agents"])]
        
        all_eval_rewards.append(sum(total_rewards))

    final_score = sum(all_eval_rewards)
    if pbar:
        pbar.set_description(f"Final Score for Trial: {final_score:.4f}")
    
    return final_score


if __name__ == '__main__':
    # This block can now be used for a single, standard run
    init_time = time.time()
    exp_config = {
        "num_agents": 2,
        "size": 4,
        "maps": 10,
        "num_states": 16 * 16 * 4,
        "epochs": 20,
        "episodes_per_epoch": 20,
        "episode_length": 16,
        "obstacle_density": 0.1,
        "save_every": 50, # Set to a high number or None to disable for normal runs
        "learning_rate": 0.01,
        "gamma": 0.99, # Added gamma here
        "epsilon_max": 1.0,
        "epsilon_min": 0.1,
        "renders": "renders/",
        "algorithm": "JALGT",
        "solution_concept": ParetoSolutionConcept # Note: IQLAgent doesn't use this
    }

    try:
        os.mkdir(exp_config["renders"])
    except FileExistsError:
        pass

    final_reward = run_experiment(exp_config)
    print(f"Final collective reward from the run: {final_reward}")
    print(f"Total execution time: {time.time() - init_time:.2f} seconds")