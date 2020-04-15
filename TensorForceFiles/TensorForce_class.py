from tensorforce import Agent
from TensorForceRL_parent import *
import numpy as np


class TensorForceClass(TensorForce_Parent):

    def __init__(self):
        super().__init__()


    def createRLagent(self):
        states_dict = {'type': 'float', 'shape': self.num_states}
        actions_dict = {'type': 'float', 'shape': self.num_actions, 'min_value': self.input_low, 'max_value': self.input_high}

        return  Agent.create(
            agent='tensorforce',
            states = states_dict,  # alternatively: states, actions, (max_episode_timesteps)
            actions = actions_dict,
            memory=10000,
            update=dict(unit='timesteps', batch_size=64),
            max_episode_timesteps= self.len_episode,
            optimizer=dict(type='adam', learning_rate=3e-4),
            policy=dict(network='auto'),
            objective='policy_gradient',
            reward_estimation=dict(horizon=20)
        )

