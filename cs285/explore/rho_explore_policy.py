"""
Implementation for algorithm 1 (perturbation-based exploration, 
without marginalizing over actions)
"""
import numpy as np
import torch
from gym import Env
from copy import copy

from cs285.policies.base_policy import BasePolicy
from cs285.critics.dqn_critic import DQNCritic
from cs285.infrastructure import pytorch_util as ptu

EPS = 1e-7
class RhoExplorePolicy(object):

    def __init__(self, critic, rho=0.03, lmbda=1, rho_sample=100):
        self.critic: DQNCritic = critic
        self.look_ahead_steps = lmbda # FIXME: currently lambda always be 1. enable multi-step later.
        self.num_perturb_samples = rho_sample
        self.perturb_margin = rho

    def get_action(self, obs, env: Env, policy: BasePolicy):
        
        self.env: Env = copy(env) # avoid mutating the environment in a destructive manner by shallow-copying.
        self.initial_state = copy(env)

        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        perturbed_obs = self.perturb(observation) # (number of perturbed states, state dimension)
        scores_of_perturbed_obs = self.step_ahead(perturbed_obs, policy=policy) # (number of perturbed states, )
        selected_obs = perturbed_obs[scores_of_perturbed_obs.argmax()] # (1, state dimension)
        if len(selected_obs.shape) == 1:
            selected_obs = selected_obs[None]
        actions = policy.get_action(selected_obs)
        # nearby_obs, nearby_qa_values = self.top_nearby_obs(collected_trajectories)
        # nearby_qa_values: np.ndarray = self.critic.qa_values(nearby_obs) 
        # greedy_actions = nearby_qa_values.argmax(axis=-1).squeeze() # (batch, # of nearby states)
        # actions = ptu.from_numpy(greedy_actions).mode(dim=-1)
        return actions

    def perturb(self, obs):
        """
        Given one observation, create `num_perturb_samples` samples given 
        the constraint of norm < `self.perturb_margin`
        """
        
        assert len(obs.shape) == 2, "obs should be 2 dimensional"
        noises = np.random.rand(obs.shape[0], obs.shape[1])
        noises = [
            [noises[d] / np.linalg.norm(noises[d]) for d in range(len(obs))]
            for _ in range(self.num_perturb_samples)
        ]
        noises = np.array(noises).squeeze() * self.perturb_margin
        if len(noises.shape) == 1:
            noises = noises[None]
        perturbed_states = np.add(obs.T, noises.T).T
        return perturbed_states

    def step_ahead(self, perturbed_obs: np.ndarray, policy: BasePolicy):
        """
        Args:
            perturbed_obs (np.ndarray): a list of perturbed states
            policy (object): a policy for mini rollout to operate on
        Return:
            A list of "cumulative reward + Q value of the last state in mini-rollout",
            ordered by the perturbed_obs. 
        """

        # FIXME: Use `gym.vector.AsyncVectorEnv` to vectorize  
        #        different perturbed states' traversal.
        
        qa_results = []
        for ob in perturbed_obs:
            mini_rollout_rewards = 0
            latest_ob = ob
            for t in range(self.look_ahead_steps):
                action = policy.get_action(latest_ob)
                latest_ob, reward, terminal, metadata = self.env.step(action)
                mini_rollout_rewards += reward
                if terminal:
                    break
                
            score = max(self.critic.qa_values(latest_ob)) + mini_rollout_rewards
            qa_results.append(score)
            self.env = copy(self.initial_state) # reset environment
            
        return np.array(qa_results)
                