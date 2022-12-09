# CS285 Deep Reinforcment Learning Project

### Perturbation-based Exploration

Hyperparameters to consider (as of Dec 9th)
| Hyperparameter            | values                |
|---------------------------|-----------------------|
|`learning_rate`            | 1e-3, 5e-4 ...        |
|`rho (perturbation margin)`|3, 5, 7, 10, 20, 30, 50% (norm) |
|`rho_sample`               | 10, 20, 30            |
|`rho_schedule`             | TBD                   |
|`lambda (look ahead steps)`| 1, 10                 |
|`sample_heuristics`        | `max`, `mode of top percentile`|
|`model-free algo`          | DQN, Soft Actor Critic (SAC)   |