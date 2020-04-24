from tensorforce import Agent
from TensorForceRL_parent import *
import numpy as np


class TensorForceDQN(TensorForce_Parent):

    def __init__(self,num_states=6, num_actions=4, load=None):
        super().__init__(num_states=num_states, num_actions=num_actions,load=load)


    def createRLagent(self, load):
        states_dict = {'type': 'float', 'shape': self.num_states}
        actions_dict = {'type': 'float', 'shape': self.num_actions, 'min_value': self.input_low, 'max_value': self.input_high}

        return Agent.create(
            agent='dqn',
            states = states_dict,  # alternatively: states, actions, (max_episode_timesteps)
            actions = actions_dict,
            memory=10000,
            exploration=0.3,
            max_episode_timesteps= self.len_episode,
        )

