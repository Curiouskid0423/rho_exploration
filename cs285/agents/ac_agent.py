from collections import OrderedDict
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicy, MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params
        
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        
        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )

        if 'exploration_schedule' in agent_params:
            self.exploration = agent_params['exploration_schedule']
            # FIXME: Add exploration to ac_agent logic
            
        self.critic = BootstrappedContinuousCritic(self.agent_params)
        if 'lander' in agent_params and agent_params['lander']:
            self.replay_buffer = MemoryOptimizedReplayBuffer(
                agent_params['replay_buffer_size'], 
                agent_params['frame_history_len'], 
                lander=agent_params['lander']
            )
        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        
        loss = OrderedDict()
        loss['Critic_Loss'] = []
        loss['Actor_Loss'] = []

        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            loss['Critic_Loss'].append(critic_loss)
            
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, adv_n=ptu.from_numpy(advantage))
            loss['Actor_Loss'].append(actor_loss)
        
        loss['Critic_Loss'] = np.array(loss['Critic_Loss']).mean()
        loss['Actor_Loss'] = np.array(loss['Actor_Loss']).mean()

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma * V(s')
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        terminal_n = ptu.from_numpy(terminal_n)
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        
        curr_val, next_val = self.critic(ob_no), self.critic(next_ob_no)
        mask = torch.where(terminal_n == 1, 0, 1)
        adv_n = re_n + self.gamma * next_val * mask - curr_val
        adv_n = ptu.to_numpy(adv_n)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
