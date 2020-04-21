from tensorforce import Agent
import numpy as np


class TensorForce_Parent:

    def __init__(self, num_states=6, num_actions=4, load=None):

        self.num_states = num_states
        self.num_actions = num_actions
        self.input_high = 1.0
        self.input_low  = 0.0
        
        
        self.len_episode = 10
        self.explore = 0.5

        self.x_r = [-0.025, 0.52]   ## X Range: -0.025 - 0.52
        self.y_r = [-0.45, 0.45]    ## Y Range: -0.45 - 0.45 
        self.z_r = [0.751, 1.75]    ## Z Range: 0.751 - 1.75 (Maybe a little higher)

        self.dist_before_action = 0
        self.dist_after_action = 0

        self.has_object = False

        self.agent = self.createRLagent(load=load)
        self.target_state = []




    def act(self, obs, obj_poses):
        gripper_pose = obs.gripper_pose


        key = 'sugar'
        ###########################################################
        ###### PREPARE INPUT STATES TO RL FUNCTION ################
        if key in obj_poses:
            target_state = list(obj_poses[key])
            target_state[2] += 0.1
        else:
            self.has_object = True
            target_state = [0.2, 0.0, 1.1]
        # in_states = list(gripper_pose)
        # in_states.extend(target_state)

        in_states = list(gripper_pose[:3])
        in_states.extend(list(target_state[:3]))
        # in_states.extend(list(obj_poses['cupboard']))
        ###### PREPARE INPUT STATES TO RL FUNCTION ################
        ###########################################################

        actions = self.agent.act(states= in_states)
        if self.explore > np.random.uniform():
            actions = np.random.uniform(low=0.25, high=0.75, size=self.num_actions)

        a_in = self.scaleActions(actions)

        actions2 = list(a_in[:3]) + [0,1,0,0] + list([actions[3]>0.5])

        self.dist_before_action = np.linalg.norm(target_state[:3] - gripper_pose[:3])
        return actions2


    def scaleActions(self, actions):
   
        actions[0] = actions[0]*(self.x_r[1] - self.x_r[0]) + self.x_r[0]
        actions[1] = actions[1]*(self.y_r[1] - self.y_r[0]) + self.y_r[0]
        actions[2] = actions[2]*(self.z_r[1] - self.z_r[0]) + self.z_r[0]
    
        return actions

    def calculateReward(self):
        terminal = False
        reward = -self.dist_before_action/4


        if self.dist_after_action < 0.2:
            reward +=  20 + 1/self.dist_after_action

        temp = (self.dist_before_action - self.dist_after_action) / self.dist_before_action * 3
        if temp > 0:
            reward += temp
        else:
            reward += min(temp,-0.1)

        

        if self.has_object: 
            reward += 100.0
            terminal = True

        

        return reward, terminal

