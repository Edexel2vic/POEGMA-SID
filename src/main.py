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
    # Use the base_seed if provided for more randomness
    actual_seed = config.get('base_seed', 42) + seed
    
    grid_config = GridConfig(num_agents=config["num_agents"],
                             size=config["size"],
                             density=config["obstacle_density"],
                             seed=actual_seed,  # Use the modified seed
                             max_episode_steps=config["episode_length"],
                             obs_radius=1,
                             on_target="finish",
                             render_mode=None)
    # Disable animations during hyperparameter search for speed
 
    animation_config = AnimationConfig(directory='renders/',
                                           static=False,
                                           show_agents=True,
                                           egocentric_idx=None,
                                           save_every_idx_episode=config["save_every"],
                                           show_border=True,
                                           show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
        
    return RewardWrapper(env)


# We wrap the main logic in a function that takes the config and returns the score
def run_experiment(config, pbar=None, trial_number=None):
    """
    Runs a single experiment with a given configuration.
    Returns detailed metrics including timing information.
    """
    experiment_start_time = time.time()
    
    # Get unique seed for this run
    run_seed = config.get('base_seed', 42)
    
    # Initialize timing tracking
    track_timing = config.get('track_timing', False)
    show_progress = config.get('show_progress', True)
    episode_training_times = []
    
    game = GameModel(num_agents=config["num_agents"], num_states=config["num_states"],
                     num_actions=5)
    
    # IMPORTANT: Use unique seeds for algorithm initialization
    if config["algorithm"] == "JALGT":
        algorithms = [JALGT(agent_id=i,
                           game=game,
                           solution_concept=config["solution_concept"](),
                           gamma=config["gamma"],
                           alpha=config["learning_rate"],
                           epsilon=config["epsilon_max"],
                           seed=run_seed + i) # Use unique seed per agent
                    for i in range(game.num_agents)]
        
    elif config["algorithm"] == "IQL":
        algorithms = [IQLAgent(agent_id=i, 
                           num_local_states=config["num_states"], 
                           num_individual_actions=5, 
                           epsilon_start=config["epsilon_max"], 
                           epsilon_end=config["epsilon_min"],
                           gamma=config["gamma"],
                           alpha=config["learning_rate"], 
                           seed=run_seed + i)  # Use unique seed per agent
                  for i in range(game.num_agents)]

    epsilon_diff = (config["epsilon_max"] - config["epsilon_min"]) / config["episodes_per_epoch"]
    
    num_epochs = config["epochs"]
    total_episodes = num_epochs * config["episodes_per_epoch"]
    episode_counter = 0
    
    if pbar:
        iterator = range(num_epochs)
    elif show_progress:
        # Using tqdm if no external progress bar is provided and progress is enabled
        desc = f"Experiment {trial_number}" if trial_number else "Running Experiment"
        iterator = tqdm(range(num_epochs), desc=desc)
    else:
        # No progress bar for parallel runs
        iterator = range(num_epochs)

    # Training phase
    for epoch in iterator:
        for ep in range(config["episodes_per_epoch"]):
            episode_start_time = time.time()
            
            if pbar: 
                pbar.set_postfix({
                    'modo': 'entrenamiento', 
                    'episodio': f"{episode_counter+1}/{total_episodes}",
                    'trial': trial_number or 'N/A'
                })
            
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
            
            # Track episode training time
            if track_timing:
                episode_time = time.time() - episode_start_time
                episode_training_times.append(episode_time)
            
            episode_counter += 1

    # Evaluation phase (after all training epochs are done)
    evaluation_episodes = config["maps"]
    all_eval_rewards = []
    individual_rewards_per_episode = []
    
    for ep in range(evaluation_episodes):
        if pbar: 
            pbar.set_postfix({
                'modo': 'evaluaciÃ³n', 
                'episodio': f"{ep+1}/{evaluation_episodes}",
                'trial': trial_number or 'N/A'
            })
        
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
        
        env.save_animation(f"{exp_config['renders']}/{'Jalgt-Welfare'}-den{ep}{exp_config['obstacle_density']}-size{exp_config['size']}-num_agents{exp_config['num_agents']}.svg",
                                   AnimationConfig(egocentric_idx=None, show_border=True, show_lines=True))
                                   
        collective_reward = sum(total_rewards)
        all_eval_rewards.append(collective_reward)
        individual_rewards_per_episode.append(total_rewards.copy())

    # Calculate final metrics
    final_collective_reward = sum(all_eval_rewards)
    mean_individual_rewards = np.mean(individual_rewards_per_episode, axis=0).tolist()
    total_training_time = time.time() - experiment_start_time
    
    if pbar and show_progress:
        pbar.set_description(f"Trial {trial_number}: Score {final_collective_reward:.2f}")
    
    # Return detailed metrics
    if track_timing:
        return {
            'collective_reward': final_collective_reward,
            'individual_rewards': mean_individual_rewards,
            'episode_training_times': episode_training_times,
            'total_training_time': total_training_time,
            'num_episodes': total_episodes,
            'evaluation_rewards': all_eval_rewards,
            'individual_rewards_per_evaluation': individual_rewards_per_episode
        }
    else:
        # For backward compatibility, return just the score if timing is not tracked
        return final_collective_reward


if __name__ == '__main__':
    # This block can now be used for a single, standard run
    for num_agents in range(2, 5):
        for size in [4, 6, 8]:
            for obs_density in [(x * (size / 4.0)) / (num_agents - 1.0) for x in [0.0, 0.1, 0.2, 0.3]]:
                init_time = time.time()
                exp_config = {
                    "num_agents": num_agents,
                    "size": size,
                    "maps": 10,
                    "num_states": 16 * 16 * 4,
                    "epochs": 10,
                    "episodes_per_epoch": 34,
                    "episode_length": 14,
                    "obstacle_density": obs_density,
                    "save_every": None, # Set to a high number or None to disable for normal runs
                    "learning_rate": 0.047225128696931,
                    "gamma": 0.9675486944707316, # Added gamma here
                    "epsilon_max": 0.9809888104530086,
                    "epsilon_min": 0.0566205652581132,
                    "renders": "renders/",
                    "algorithm": "JALGT",
                    "solution_concept": WelfareSolutionConcept, # Note: IQLAgent doesn't use this
                    "track_timing": True  # Enable detailed timing for standalone runs
                }

                try:
                    os.mkdir(exp_config["renders"])
                except FileExistsError:
                    pass

                result = run_experiment(exp_config)
                
                if isinstance(result, dict):
                    print(f"PARAMS ======================================")
                    print(f"NUM_AGENTS:  {num_agents}, SIZE: {size}, OBSTACLE: {obs_density}")
                    print(f"Final collective reward: {result['collective_reward']}")
                    print(f"Individual rewards (mean): {result['individual_rewards']}")
                    print(f"Total training time: {result['total_training_time']:.2f} seconds")
                    print(f"Average time per episode: {result['total_training_time']/result['num_episodes']:.4f} seconds")
                    print(f"Total episodes: {result['num_episodes']}")
                else:
                    print(f"Final collective reward: {result}")
                
                print(f"Total execution time: {time.time() - init_time:.2f} seconds")               
