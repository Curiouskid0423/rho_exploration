from collections import namedtuple
import torch.optim as optim
from torch import nn
from cs285.infrastructure.atari_wrappers import wrap_deepmind
from cs285.infrastructure.dqn_utils import LinearSchedule, PiecewiseSchedule

OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)

def get_atari_env_kwargs(env_name):

    """
    Feasible set of hyperparameters adapted from CS285 code.
    """
    
    if env_name in ['MsPacman-v0', 'PongNoFrameskip-v4']:
        kwargs = {
            'learning_starts': 50000,
            'critic_target_update_frequency': 10000,
            'replay_buffer_size': int(1e6),
            'num_timesteps': int(2e8),
            'grad_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'frame_history_len': 4,
            # 'q_func': create_atari_q_network,
            # 'gamma': 0.99,
        }
        kwargs['optimizer_spec'] = atari_optimizer(kwargs['num_timesteps'])
        kwargs['exploration_schedule'] = atari_exploration_schedule(kwargs['num_timesteps'])

    elif env_name == 'LunarLander-v3':
        def lunar_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': lander_optimizer(),
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'learning_starts': 1000,
            'frame_history_len': 1,
            'critic_target_update_frequency': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 300000,
            'env_wrappers': lunar_empty_wrapper,
            # 'gamma': 1.00,
            'q_func': create_lander_q_network,
        }
        kwargs['exploration_schedule'] = LinearSchedule(kwargs['num_timesteps'], final_p=0., initial_p=.5)

    else:
        raise NotImplementedError

    return kwargs


def create_lander_q_network(ob_dim, num_actions):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )

def lander_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=lambda epoch: 1e-3,  # keep init learning rate
        # learning_rate_schedule=lambda epoch: 5e-4,  # keep init learning rate
    )

def atari_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_optimizer(num_timesteps):
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-1),
            (num_timesteps / 40, 1e-1),
            (num_timesteps / 8, 5e-2),
        ],
        outside_value=5e-2,
    )

    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1e-3,
            eps=1e-4
        ),
        learning_rate_schedule=lambda t: lr_schedule.value(t),
    )

