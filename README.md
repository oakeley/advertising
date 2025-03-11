# Digital Advertising Optimization with Reinforcement Learning

This repository contains a reinforcement learning (RL) framework designed to optimize digital advertising strategies using Deep Q-Learning. The system learns optimal bidding strategies by simulating advertising campaigns and making decisions based on various advertising metrics.

## Project Overview

The system uses Deep Q-Networks (DQN) to learn when to adopt either conservative or aggressive advertising strategies based on keyword characteristics and campaign performance. It analyzes factors such as:

- Keyword competitiveness
- Organic ranking and performance
- Paid click metrics
- Conversion rates and ROAS (Return on Ad Spend)
- Cost per click and acquisition

## Features

- **Synthetic Data Generation**: Creates realistic advertising data with appropriate correlations between metrics
- **RL Environment**: Simulates a digital advertising environment with state transitions and rewards
- **Deep Q-Network**: Neural network model for learning advertising optimization policies
- **Experience Replay**: Memory buffer for more efficient and stable learning
- **Epsilon-Greedy Exploration**: Balance between exploring new strategies and exploiting known effective ones
- **Comprehensive Evaluation**: Detailed metrics and visualizations of model performance

## Sample Output

```
Using device: cuda
Starting digital advertising optimization pipeline...
Results will be saved to: ad_optimization_results_20250311_112735
Generating synthetic dataset...
Synthetic dataset saved to ad_optimization_results_20250311_112735/synthetic_ad_data.csv
Dataset summary:
Shape: (1000, 17)
Feature stats:
       competitiveness  difficulty_score  organic_rank  organic_clicks  organic_ctr  paid_clicks     paid_ctr     ad_spend  ad_conversions      ad_roas  conversion_rate  cost_per_click
count      1000.000000       1000.000000   1000.000000     1000.000000  1000.000000    1000.0000  1000.000000  1000.000000     1000.000000  1000.000000      1000.000000     1000.000000
mean          0.418282          0.418907      4.286000      125.000000     0.026165      63.3740     0.026496   260.633521        4.203000     1.137278         0.077270        4.452296
std           0.206529          0.152807      1.687321      191.484491     0.018377     179.9339     0.020284   649.355848       15.083212     1.277420         0.062721        2.113999
min           0.008723          0.039553      1.000000        1.000000     0.010000       0.0000     0.010000     0.000000        0.000000     0.500000         0.010000        0.573723
25%           0.257963          0.308226      3.000000       26.000000     0.010328       9.0000     0.011712    32.123617        0.000000     0.500000         0.028243        2.833476
50%           0.407154          0.414651      4.000000       60.000000     0.020227      22.0000     0.018457    88.178336        1.000000     0.500000         0.060762        4.278748
75%           0.574862          0.528856      6.000000      144.000000     0.034871      56.2500     0.034998   241.174143        3.000000     1.083987         0.106583        5.701865
max           0.913124          0.839623      9.000000     2018.000000     0.114515    3069.0000     0.155394  7721.303514      277.000000     5.000000         0.300000       10.000000
Training RL agent...
Starting training...
Episode 10/200, Avg Reward: -471.75, Epsilon: 0.95
...
Episode 200/200, Avg Reward: -269.10, Epsilon: 0.37
Training completed!
...
Pipeline completed successfully. All results saved to ad_optimization_results_20250311_112735
```

## Project Structure

The main script `Digital_Advertising_Env_Roger_Edward.py` contains:

- Data generation functions
- Environment definition
- Neural network architecture
- Training and evaluation pipeline
- Visualization utilities

## Getting Started

### Prerequisites

The project requires Python 3.13 and several libraries including PyTorch, NumPy, Pandas, Matplotlib, and Seaborn.

### Environment Setup

Create a conda environment with Python 3.13:

```bash
# Create a new conda environment
conda create -n ad-optimization python=3.13
conda activate ad-optimization

# Install required packages
conda install -c pytorch pytorch
conda install numpy pandas matplotlib seaborn
conda install -c conda-forge tqdm ipykernel
```

Or use the environment.yml file:

```bash
# Create environment from file
conda env create -f environment.yml
conda activate ad-optimization
```

### Environment YAML

```yaml
name: ad-optimization
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.13
  - pytorch
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - ipykernel
  - tqdm
  - pip
  - pip:
    - torchvision
```

## Usage

Run the main script:

```bash
python Digital_Advertising_Env_Roger_Edward.py
```

The script will:
1. Generate synthetic advertising data
2. Train the RL agent
3. Evaluate the trained model
4. Create visualizations of training progress and performance
5. Save the trained model and all outputs to a timestamped directory

## Output Files

The program creates a timestamped directory containing:
- Synthetic dataset (CSV)
- Trained model (PyTorch)
- Evaluation metrics (text file)
- Training progress visualization
- Evaluation visualization

## Key Components

### AdEnv Class
Simulates the digital advertising environment, providing observations and rewards based on keyword characteristics and actions taken.

### QNetwork Class
Neural network architecture for approximating the Q-function, predicting the expected future rewards for each action.

### DQNAgent Class
Implements the reinforcement learning agent with policy and target networks, epsilon-greedy exploration, and training logic.

### ReplayBuffer Class
Stores agent experiences (state, action, reward, next state, done) for more efficient and stable learning.

## Customization

You can customize the project by:
- Adjusting the `feature_columns` list to include different metrics
- Modifying the reward function in `_compute_reward` to align with specific business objectives
- Tuning hyperparameters like learning rate, epsilon decay, or network architecture
- Generating more synthetic data samples for more robust training

## License

This project is licensed under the GPL v3. See the file LICENSE for details.

## Acknowledgments

- This project uses techniques from Deep Q-Learning literature
- The synthetic data generation is designed to mimic real-world digital advertising patterns
