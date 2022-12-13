import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure.sac_utils import soft_update_params
import cs285.infrastructure.pytorch_util as ptu
from cs285.explore.rho_explore_policy import RhoExplorePolicy
from gym.spaces import Discrete

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env

        if isinstance(self.env.action_space, Discrete):
            vec = self.env.action_space
            self.action_range = [
                vec.start, 
                vec.start+vec.n-1
            ]
        else:
            self.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]

        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']
        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

        if 'rho_explore' in self.agent_params and self.agent_params['rho_explore']:
            self.rho_explorer = RhoExplorePolicy(
                critic = self.critic,
                rho = agent_params['rho'], 
                lmbda = agent_params['lambda'], 
                rho_sample = agent_params['rho_sample'],
                sample_heuristics = agent_params['heuristics']
            )
            self.mean_state_norm = None

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        
        with torch.no_grad():
            # get next action
            next_action_dist = self.actor(next_ob_no)
            sampled_action = next_action_dist.sample() # could be rank-1 in `discrete` env
            
            # Convert discrete actions to one-hot vectors
            if isinstance(self.env.action_space, Discrete):
                next_action = F.one_hot(sampled_action)
            else:
                next_action = sampled_action
            
            # compute Q value of the next state
            next_Qs = self.critic_target(next_ob_no, next_action)
            next_Q = torch.min(*next_Qs)

            # compute target Q (of the next state)
            target_Q = re_n + ((1-terminal_n) * self.gamma * next_Q)
            next_log_prob = next_action_dist.log_prob(sampled_action).sum(-1, keepdim=True)
            target_Q -= self.gamma * (1-terminal_n) * self.actor.alpha.detach() * next_log_prob

        critic_loss = 0

        # get current Q estimates
        current_Qs = self.critic(ob_no, ac_na)
        for current_Q in current_Qs:
            critic_loss += self.critic.loss(current_Q, target_Q)

        # optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n).unsqueeze(1)
        terminal_n = ptu.from_numpy(terminal_n).unsqueeze(1)
        
        if self.actor.discrete:
            ac_na = F.one_hot(ac_na.to(torch.int64))
        assert ac_na.shape[-1] == self.agent_params['ac_dim']

        loss = OrderedDict()

        # update the critic network (Q function, `theta` in the paper)
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            loss['Critic_Loss'] = critic_loss
            if self.training_step % self.critic_target_update_frequency == 0:
                soft_update_params(self.critic, self.critic_target, self.critic_tau)

        # periodically update the Actor network
        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, temperature = self.actor.update(ob_no, self.critic)
                loss['Actor_Loss'] = actor_loss
                loss['Alpha_Loss'] = alpha_loss
                loss['Temperature'] = temperature

        self.training_step += 1

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
