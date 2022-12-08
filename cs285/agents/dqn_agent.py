import numpy as np
from gym import Env

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic
# final project code
from cs285.explore.rho_explore_policy import RhoExplorePolicy


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env: Env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

        if 'rho_explore' in self.agent_params and self.agent_params['rho_explore']:
            self.rho_explorer = RhoExplorePolicy(
                critic = self.critic,
                rho = agent_params['rho'], 
                lmbda = agent_params['lambda'], 
                rho_sample = agent_params['rho_sample'],
                sample_heuristics = agent_params['heuristics']
            )
            self.mean_state_norm = None       

    def add_to_replay_buffer(self, paths):
        pass

    def determine_policy(self):
        """
        A helper method to decide between random exploration, 
        local exploration, or exploitation.
        """
        eps = self.exploration.value(self.t)
        cap = .5
        count = np.random.random()
        if not hasattr(self, 'rho_explorer'):
            if (count > (1 - eps)) or self.t < self.learning_starts:
                return 'sparse'
            else:
                return 'exploit'
        else:
            if count > cap: # exploitation
                return 'exploit'
            else: # exploration
                encourage_sparcity = eps > cap or count < eps # eps typically decreases over time
                if encourage_sparcity or self.t < self.learning_starts:
                    return 'sparse'
                else:
                    return 'local'

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer` in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        # TODO use epsilon greedy exploration when selecting action
        # eps = self.exploration.value(self.t)
        # perform_random_action = (np.random.random() > (1 - eps)) or self.t < self.learning_starts
        
        explore_or_exploit = self.determine_policy()
        
        if explore_or_exploit == 'sparse':
            action = self.env.action_space.sample()
        elif explore_or_exploit == 'local':
            recent_obs = self.replay_buffer.encode_recent_observation()
            if self.mean_state_norm is None:
                self.compute_mean_state_norm()
            action = self.rho_explorer.get_action(
                obs=recent_obs, 
                env=self.env, 
                policy=self.actor,
                perturb_unit=self.mean_state_norm
            )
        else:
            # Actor will take in multiple previous observations ("frames") to deal with the 
            # partial observability of the environment. Get the most recent `frame_history_len` 
            # observations using functionality from the replay buffer, and then use those 
            # observations as input to your actor. 
            recent_obs = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(recent_obs)
        
        obs, reward, terminal, metadata = self.env.step(action)
        self.last_obs = obs

        # TODO store the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done=terminal)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if terminal:
            self.last_obs = self.env.reset()

    def compute_mean_state_norm(self, n=1000):
        
        from numpy.linalg import norm
        
        print("Calculating the mean of state vector norm...")
        obs, act, _, _, _ = self.sample(
            batch_size=min(n, self.replay_buffer.size))
        assert isinstance(obs, np.ndarray) and len(obs.shape) == 2
        obs_norms = norm(obs, axis=-1, keepdims=False)
        assert len(obs_norms.shape) == 1
        self.mean_state_norm = obs_norms.mean(axis=0)
        print(f"mean_state_norm (sample size {n}): {self.mean_state_norm}\n")
        
    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
