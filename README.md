# Neighboring States-based RL Exploration

All training scripts are located in the `scripts/` directory. To be updated with more extensive experiments on different agents (e.g. SAC, PPO).

Hyperparameters to consider (as of Dec 2022)
| Hyperparameter            | values                |
|---------------------------|-----------------------|
|`learning_rate`            | 1e-3, 5e-4            |
|`rho (perturbation margin)`| 3, 5, 7, 10, 20%      |
|`rho_sample`               | 10, 20, 30            |
|`lambda (look ahead steps)`| 1, 10                 |
|`sample_heuristics`        | `max`, `mode of top percentile`|
|`model-free algo`          | DQN (Actor Critic, SAC, PPO)   |