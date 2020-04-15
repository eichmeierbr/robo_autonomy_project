from tensorforce import Agent
import numpy as np


class TensorForceDQN:

    def __init__(self):

        self.num_states = 6
        self.num_actions = 4
        self.input_high = 1.0
        self.input_low  = 0.0
        
        states_dict = {'type': 'float', 'shape': self.num_states}
        actions_dict = {'type': 'float', 'shape': self.num_actions, 'min_value': self.input_low, 'max_value': self.input_high}
        
        self.len_episode = 15
        self.explore = 0.3


        self.agent = Agent.create(
            agent='dqn',
            states = states_dict,  # alternatively: states, actions, (max_episode_timesteps)
            actions = actions_dict,
            memory=10000,
            max_episode_timesteps= self.len_episode,
        )


        self.x_r = [-0.0, 0.45]   ## X Range: -0.025 - 0.52
        self.y_r = [-0.4, 0.4]    ## Y Range: -0.45 - 0.45 
        self.z_r = [0.7, 1.65]    ## Z Range: 0.751 - 1.75 (Maybe a little higher)

        self.dist_before_action = 0
        self.dist_after_action = 0

        self.has_object = False


    def act(self, obs, obj_poses):
        gripper_pose = obs.gripper_pose

        ###########################################################
        ###### PREPARE INPUT STATES TO RL FUNCTION ################
        target_state = list(obj_poses['sugar'])
        target_state[2] += 0.1
        # in_states = list(gripper_pose)
        # in_states.extend(target_state)

        in_states = list(gripper_pose[:3])
        in_states.extend(list(target_state[:3]))
        # in_states.extend(list(obj_poses['cupboard']))
        ###### PREPARE INPUT STATES TO RL FUNCTION ################
        ###########################################################

        actions = self.agent.act(states= in_states)
        if self.explore > np.random.uniform():
            actions = np.random.uniform(size=self.num_actions)

        a_in = self.scaleActions(actions)

        actions2 = list(a_in) + [0,1,0,0] + list([actions[3]>0.5])

        self.dist_before_action = np.linalg.norm(target_state[:3] - gripper_pose[:3])
        return actions2


    def scaleActions(self, actions):
   
        actions[0] = actions[0]*(self.x_r[1] - self.x_r[0]) + self.x_r[0]
        actions[1] = actions[1]*(self.y_r[1] - self.y_r[0]) + self.y_r[0]
        actions[2] = actions[2]*(self.z_r[1] - self.z_r[0]) + self.z_r[0]
    
        return actions

    def calculateReward(self):
        terminal = False
        reward = 0.0

        # If our action took us closer to the goal
        if self.dist_before_action > self.dist_after_action:
            reward = 1/self.dist_after_action
            # best_reward = max(reward, best_reward)

        elif self.dist_before_action < self.dist_after_action:
            reward = 0.0

        if self.has_object: 
            reward = 50.0
            terminal = True

        return reward, terminal





