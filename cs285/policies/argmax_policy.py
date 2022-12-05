import numpy as np

from cs285.critics.dqn_critic import DQNCritic


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic: DQNCritic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maximizes the Q-value 
        # at the current observation as the output
        qa_values: np.ndarray = self.critic.qa_values(observation)
        greedy_actions = qa_values.argmax(axis=-1)
        return greedy_actions.squeeze()