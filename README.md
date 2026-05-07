**Note:** This research and the original documentation were conducted in Spanish as part of the **Distributed Intelligent Systems (SID)** course at the **Faculty of Informatics of Barcelona (FIB) - UPC**
# POEGMA-SID

Multi-Agent Reinforcement Learning (MARL) implementation for the POGEMA (Partially Observable Grid Environment for Multiple Agents) environment. This project implements and compares various solution concepts and learning algorithms for cooperative pathfinding in multi-agent systems.

## Overview

This project explores multi-agent reinforcement learning in grid-based pathfinding scenarios where multiple agents must navigate to their respective goals while avoiding obstacles and other agents. The implementation includes several MARL algorithms and solution concepts to coordinate agent behavior.

### Key Features

- **Multiple MARL Algorithms**:
  - **JALGT** (Joint Action Learning with Game Theory): A cooperative learning algorithm that uses game-theoretic solution concepts
  - **IQL** (Independent Q-Learning): Agents learn independently without explicit coordination
  - **Neural Network-based approaches**: Deep learning implementations for agent policies

- **Game-Theoretic Solution Concepts**:
  - **Pareto Optimality**: Finds Pareto-efficient joint actions
  - **Nash Equilibrium**: Computes Nash equilibrium strategies
  - **Minimax**: Minimizes maximum opponent reward
  - **Welfare Maximization**: Maximizes collective welfare

- **Hyperparameter Optimization**: Integration with Optuna for automated hyperparameter tuning

- **Visualization**: Generates SVG animations of agent trajectories and learning progress

## Installation

### Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Androm3d/POEGMA-SID.git
cd POEGMA-SID
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- `numpy`, `pandas` - Data manipulation
- `gymnasium` - RL environment interface
- `pogema` - POGEMA environment
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `optuna` - Hyperparameter optimization
- `scikit-learn` - Machine learning utilities
- `tqdm` - Progress bars
- `numba` - Performance optimization

## Usage

### Basic Training

Run the main training script with default configuration:

```bash
python main.py
```

### Training with Optuna Hyperparameter Optimization

For automated hyperparameter tuning:

```bash
cd src
python main_optuna.py
```

This will run multiple trials with different hyperparameter configurations and save the results to a database.

### Jupyter Notebook

Explore the baseline implementation interactively:

```bash
jupyter notebook baseline.ipynb
```

## Project Structure

```
POEGMA-SID/
├── main.py                 # Main training script (root level)
├── src/                    # Source code directory
│   ├── main.py            # Alternative main training script
│   ├── main_optuna.py     # Hyperparameter optimization with Optuna
│   ├── algorithms.py      # MARL algorithm implementations (JALGT, IQL)
│   ├── solution_concepts.py # Game-theoretic solution concepts
│   ├── game_model.py      # Game model and action space
│   ├── utils.py           # Utility functions for visualization
│   ├── best_params.py     # Best hyperparameters from optimization
│   ├── trial.py           # Trial execution utilities
│   └── graficos1.py       # Plotting utilities
├── redNeuronal/           # Neural network-based implementations
├── baseline.ipynb         # Jupyter notebook with baseline examples
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Configuration

The main training script uses a configuration dictionary with the following key parameters:

### Environment Configuration

- `num_agents`: Number of agents in the environment (default: 2)
- `size`: Grid size (width and height) (default: 4)
- `obstacle_density`: Probability of obstacles in the grid (default: 0.1)
- `episode_length`: Maximum steps per episode (default: 16)

### Training Configuration

- `epochs`: Number of training epochs (default: 200)
- `episodes_per_epoch`: Episodes per training epoch (default: 10)
- `maps`: Number of different maps for training/evaluation (default: 10)

### Algorithm Configuration

- `learning_rate` (alpha): Learning rate for Q-table updates (default: 0.01)
- `epsilon_max`: Initial exploration rate (default: 1.0)
- `epsilon_min`: Minimum exploration rate (default: 0.1)
- `num_states`: Total number of possible states (default: 16*16*4 = 1024)
  - Based on: Obstacles (4 bits) × Agents (4 bits) × Target (2 bits)

### Solution Concept

Choose from available solution concepts:
```python
solution_concept = ParetoSolutionConcept  # or MinimaxSolutionConcept, 
                                          # NashSolutionConcept, 
                                          # WelfareSolutionConcept
```

## State Representation

The state is encoded as a single integer combining:

1. **Target Direction** (2 bits, values 0-3):
   - 0: Target is up or diagonal up
   - 1: Target is down or diagonal down
   - 2: Target is left (not diagonal)
   - 3: Target is right (not diagonal)

2. **Agent Positions** (4 bits, positions 2-5):
   - Bit 5 (2^5): Agent above
   - Bit 4 (2^4): Agent to the left
   - Bit 3 (2^3): Agent to the right
   - Bit 2 (2^2): Agent below

3. **Obstacles** (4 bits, positions 6-9):
   - Bit 9 (2^9): Obstacle above
   - Bit 8 (2^8): Obstacle to the left
   - Bit 7 (2^7): Obstacle to the right
   - Bit 6 (2^6): Obstacle below

## Solution Concepts

### Pareto Optimality
Selects joint actions that are Pareto-efficient, meaning no agent can improve without making another agent worse off. Distributes probability uniformly across all Pareto-optimal joint actions.

### Nash Equilibrium
Finds joint actions where no agent can unilaterally improve their reward. Computes best responses for all agents and identifies mutual best responses.

### Minimax
Each agent minimizes the maximum reward the opponent can achieve. Uses softmax over negative opponent values to compute action probabilities.

### Welfare Maximization
Selects joint actions that maximize the sum of all agents' rewards, promoting collective cooperation.

## Algorithms

### JALGT (Joint Action Learning with Game Theory)

- Maintains a joint Q-table for all agents
- Updates Q-values based on joint actions and rewards
- Uses game-theoretic solution concepts to compute joint policies
- Supports epsilon-greedy exploration with linear decay

### IQL (Independent Q-Learning)

- Each agent maintains its own Q-table based on local observations
- Agents learn independently without coordination
- Uses standard Q-learning updates with epsilon-greedy exploration
- Optimized with Numba JIT compilation for performance

## Output

The training process generates:

1. **Learning Curves**: Plots of collective reward and TD error over epochs
2. **Animations**: SVG files showing agent trajectories for each epoch
   - Saved in `renders/` directory
   - Format: `{SolutionConcept}-map{id}-agent{id}-epoch{id}.svg`
3. **Metrics**: Training metrics including rewards and TD errors

## Performance Optimization

The codebase uses several optimization techniques:

- **Numba JIT compilation**: Critical functions are JIT-compiled for speed
- **Parallel processing**: Support for parallel trial execution in hyperparameter optimization
- **Vectorization**: NumPy operations for efficient array computations
- **Caching**: Function result caching where applicable

## Research Findings
The complete comparative study, detailed performance analysis, and hyperparameter optimization results are available in the full project report: MARL-POGEMA.pdf.  Note that the comprehensive report is written in Spanish, reflecting its origin as an academic project for the Distributed Intelligent Systems (SID) course at FIB - UPC.

### Key Takeaways:
* **Best Performance**: **JALGT-NN** with **Pareto Optimality** proved to be the most robust configuration for coordination.
* **Scalability**: While joint-action methods excel in coordination, **IQL** is more efficient as the number of agents increases due to lower computational complexity.
* **Hyperparameters**: The most critical parameters (like `episode_length`) vary significantly depending on the chosen solution concept.

## Contributing

This project appears to be an academic assignment or research project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Authors

* **Víctor Ramírez Arimaha** 
* **Marcel Alabart Benoit** 
* **Adrià Cebrián Ruiz** 
* *Facultad de Informática de Barcelona (FIB) - Universidad Politécnica de Cataluña (UPC)*

## License

Please check with the repository owner for licensing information.

## Acknowledgments

- Built using the [POGEMA](https://github.com/AIRI-Institute/pogema) environment
- Implements algorithms and concepts from multi-agent reinforcement learning research
- Uses [Optuna](https://optuna.org/) for hyperparameter optimization

## Citation

If you use this code in your research, please cite appropriately and check with the original authors.
