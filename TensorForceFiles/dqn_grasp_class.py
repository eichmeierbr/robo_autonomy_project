from tensorforce import Agent
from TensorForceRL_parent import *
from DQN_class import *
import numpy as np


class DQN_grasp(TensorForceDQN):

    def __init__(self, num_actions=3, num_states=14, load=None):
        self.num_states = num_states # Gripper pose, object pose
        self.num_actions = num_actions # X, Y, Z, Yaw, Grasp
        super().__init__(num_states= self.num_states, num_actions=self.num_actions, load=load)
        
        self.x_r = [-.001, .001] 
        self.y_r = [-.001, .001]
        self.z_r = [0.752, 1.0] ## Z Range: 0.751 - 1.75 (Maybe a little higher)
        self.yaw_r = [0, np.pi]
        self.gripper_open = True
        self.target_start_pose = [0,0,0]
        self.ee_pos = [0,0,0]
        self.explore = 0.5
        self.target_num = 0
        self.target_name=''

        

    def act(self, obs, obj_poses, key='sugar'):
        gripper_pose = obs.gripper_pose
        self.ee_pos = gripper_pose
        ###########################################################
        ###### PREPARE INPUT STATES TO RL FUNCTION ################
        if key in obj_poses:
            target_state = list(obj_poses[key])
            self.has_object = False
        else:
            self.has_object = True
            target_state = gripper_pose
            target_state[3] +=0.1

        in_states = list(gripper_pose)
        in_states.extend(list(target_state))
        ###### PREPARE INPUT STATES TO RL FUNCTION ################
        ###########################################################

        actions = self.agent.act(states=in_states)

        if self.explore > np.random.uniform():
            actions = np.random.uniform(low=0.0, high=1, size=self.num_actions)

        a_in = self.scaleActions(actions)
        self.gripper_open = a_in[-1]>0.5

        if self.num_actions == 5:
            a_in[:2] += target_state[:2]
            # a_in[:3] = gripper_pose[:3]
            self.ee_pos = a_in[:3]
            actions2 = list(self.ee_pos) + self.calculateQuaternion(a_in[3]) + list([self.gripper_open])
        
        elif self.num_actions == 3:
            self.ee_pos = [target_state[0], target_state[1], a_in[0]]
            actions2 = list(self.ee_pos) + self.calculateQuaternion(a_in[1]) + list([self.gripper_open])


        self.dist_before_action = max(0.05,np.linalg.norm(target_state[:3] - gripper_pose[:3]))
        return actions2


    def scaleActions(self, actions):
   
        if self.num_actions == 5:
            actions[0] = actions[0]*(self.x_r[1] - self.x_r[0]) + self.x_r[0]
            actions[1] = actions[1]*(self.y_r[1] - self.y_r[0]) + self.y_r[0]
            actions[2] = actions[2]*(self.z_r[1] - self.z_r[0]) + self.z_r[0]
            actions[3] = actions[3]*(self.yaw_r[1] - self.yaw_r[0]) + self.yaw_r[0]
        else:
            actions[0] = actions[0]*(self.z_r[1] - self.z_r[0]) + self.z_r[0]
            actions[1] = actions[1]*(self.yaw_r[1] - self.yaw_r[0]) + self.yaw_r[0]

        if self.has_object: actions[-1] = 0

        return actions
    

    def calculateReward(self):
        reward = 0
        terminal = False

        delta_dist = self.dist_before_action - self.dist_after_action
        temp = (self.dist_before_action - self.dist_after_action) / self.dist_before_action * 3

        if delta_dist > 0:
            reward = temp
        else:
            reward += min(temp,-0.1)
        
        if self.has_object: 
            reward += 20
            if self.ee_pos[-1] > self.z_r[1] - 0.05:
                reward += 250
                terminal = True

        if not self.gripper_open and not self.has_object:
            reward -=1

        return reward, terminal


    def calculateQuaternion(self, angle):
        firstElement  = np.sin(angle/2)
        secondElement = -np.cos(angle/2)
        return [firstElement, secondElement, 0, 0]



