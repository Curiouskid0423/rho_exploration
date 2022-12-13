from .base_critic import BaseCritic
import torch
from torch import nn
from torch import optim

from cs285.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        if 'optimizer_spec' in hparams:
            # this if clause is originally to accommodate Lunar-Lander
            self.optimizer_spec = hparams['optimizer_spec']
            self.optimizer = self.optimizer_spec.constructor(
                self.critic_network.parameters(),
                **self.optimizer_spec.optim_kwargs
            )
            self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                self.optimizer_spec.learning_rate_schedule,
            )
        else:
            self.optimizer = optim.Adam(
                self.critic_network.parameters(),
                self.learning_rate,
            )

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward

        terminal_n = ptu.from_numpy(terminal_n)
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        
        total_grad_steps = self.num_grad_steps_per_target_update * self.num_target_updates

        loss = {}
        loss['BootstrappedContinuousCritic'] = []
        target = None
        mask = torch.where(terminal_n == 1, 0, 1)
        for gs in range(total_grad_steps):
            self.optimizer.zero_grad()
            # Update the target network
            if gs % self.num_grad_steps_per_target_update == 0:
                next_state_critic_val = self.forward(next_ob_no)
                next_state_critic_val = next_state_critic_val.detach()
                target = reward_n + self.gamma * next_state_critic_val * mask
            # Update the critic (value function)
            # FIXME
            critic_val = self.forward(ob_no)
            curr_loss = self.loss(critic_val, target).mean()
            loss['BootstrappedContinuousCritic'].append(curr_loss.item())
            curr_loss.backward()
            self.optimizer.step()
            
        # Just record the latest loss value
        return loss['BootstrappedContinuousCritic'][-1]
