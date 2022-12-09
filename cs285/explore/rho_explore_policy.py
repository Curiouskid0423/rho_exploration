"""
Implementation for algorithm 1 (perturbation-based exploration, 
without marginalizing over actions)
"""
import numpy as np
from scipy import stats
import torch
from gym import Env
from gym.vector import AsyncVectorEnv
from copy import copy

from cs285.policies.base_policy import BasePolicy
from cs285.critics.dqn_critic import DQNCritic
from cs285.infrastructure import pytorch_util as ptu

EPS = 1e-7
class RhoExplorePolicy(object):

    def __init__(self, critic, rho=0.03, lmbda=1, rho_sample=100, sample_heuristics='max'):
        self.critic: DQNCritic = critic
        self.look_ahead_steps = lmbda # FIXME: currently lambda always be 1. enable multi-step later.
        self.num_perturb_samples = rho_sample
        self.perturb_margin = rho
        self.sample_heuristic = sample_heuristics # options: 'max', 'mode' (mode of top K percentile)
        self.sample_threshold = .5
        # AsyncVectorEnv config
        self.use_parallel_envs = False
        self.num_envs = 10

    def get_action(self, obs, env: Env, policy: BasePolicy, perturb_unit: float):
        
        self.env: Env = copy(env) # avoid mutating the environment in a destructive manner by shallow-copying.
        self.initial_state = copy(env)

        if self.use_parallel_envs:
            self.env = self.get_async_envs(self.env, num=self.num_envs)

        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        perturbed_obs = self.perturb(observation, perturb_unit) # (number of perturbed states, state dimension)
        scores_of_perturbed_obs = self.step_ahead(perturbed_obs, policy=policy) # (number of perturbed states, )
        actions = self.sample_by_heuristic(
            obs = perturbed_obs,
            scores = scores_of_perturbed_obs,
            heuristic = self.sample_heuristic,
            policy = policy
        ) # (1, action dimension)
        
        return actions

    def perturb(self, obs, unit: float):
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
        noises = np.array(noises).squeeze() * unit.squeeze() * self.perturb_margin
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


    def sample_by_heuristic(self, obs, scores: np.ndarray, heuristic: str, policy: BasePolicy):
        assert isinstance(scores, np.ndarray) and len(scores.shape) == 1
        if heuristic == 'max':
            selected_obs = obs[scores.argmax()]   
        elif heuristic == 'mode':
            assert hasattr(self, 'sample_threshold') 
            pool_size = max(1, int(self.sample_threshold * len(scores)))
            indices = scores.argsort()[-pool_size:] # ascending order
            selected_obs = obs[indices] # (number of indices, state dimension)
        else:
            raise NotImplementedError("heuristic not identified") 
        
        if len(selected_obs.shape) == 1:
            selected_obs = selected_obs[None]
        actions = policy.get_action(selected_obs)

        if heuristic == 'mode':
            assert len(actions.shape) == 1, "actions should be a rank-1 array"
            actions = stats.mode(actions, keepdims=False).mode
            return actions
        else:
            return actions

    def get_async_envs(self, template: Env, num=1) -> AsyncVectorEnv:
        """
        Create multiple asynchronous environments to explore in parallel.
        """

        env_list = AsyncVectorEnv([
                lambda: copy(template)
                for _ in range(num)
            ]
        )
        return env_list
        