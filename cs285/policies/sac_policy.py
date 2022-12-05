from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure.sac_utils import SquashedNormal
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        # `alpha` -- entropy temperature
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        # Actually, just return the temperature
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        
        obs = obs if len(obs.shape) > 1 else obs[None]
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs)
        
        if sample:
            action = self(obs).rsample()
        else:
            action = self(obs).mean.cpu()
        action = ptu.to_numpy(action)
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        assert not self.discrete, "Discrete mode not implemented for this homework"
        batch_mean = self.mean_net(observation)
        truncated = torch.clip(input=self.logstd, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        squashed_distribution = SquashedNormal(batch_mean, scale=torch.exp(truncated))
        return squashed_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        
        # optimize policy "actor"
        self.optimizer.zero_grad()
        
        action_distribution = self(obs)
        sampled_action = action_distribution.rsample()
        curr_action_log_prob = action_distribution.log_prob(sampled_action)
        
        critic_1, critic_2 = critic(obs, sampled_action)
        min_critic = torch.minimum(critic_1, critic_2)
        
        actor_loss = self.alpha.detach() * curr_action_log_prob - min_critic 
        actor_loss = actor_loss.mean()
        
        actor_loss.backward()
        self.optimizer.step()

        # optimize such that the entropy term converges to the negative action space dimension
        self.log_alpha_optimizer.zero_grad() # FIXME
        alpha_loss = -self.alpha * (curr_action_log_prob.detach() + self.target_entropy)
        alpha_loss = alpha_loss.mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha