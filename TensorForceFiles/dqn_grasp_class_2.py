from tensorforce import Agent
from TensorForceRL_parent import *
from DQN_class import *
from dqn_grasp import *
import numpy as np


class DQN_grasp_class_2(DQN_grasp):

    def __init__(self, num_states=16, num_actions=3, load=None):
        self.num_states = 16 # Gripper pose, object pose, Gripper Open, Target_num
        self.num_actions = 3 # Z, Yaw, Grasp
        super().__init__(num_states=num_states, num_actions=num_actions, load=load)
        
        self.z_r = [0.752, 1.0] ## Z Range: 0.751 - 1.75 (Maybe a little higher)
        self.explore = 0.3

        

    def getInStates(self,key, obj_poses):
        if key in obj_poses:
            target_state = list(obj_poses[key])
            self.target_state = target_state
            self.has_object = False
        else:
            self.has_object = True
            self.target_state = self.ee_pos
            self.target_state[3] +=0.1

        in_states = list(self.ee_pos)
        in_states.extend(list(self.target_state))

        in_states.extend(list([self.gripper_open*1.0, self.target_num]))
        return in_states

        

    def act(self, obs, obj_poses, key='sugar'):
        gripper_pose = obs.gripper_pose
        self.ee_pos = gripper_pose

        in_states = self.getInStates(key, obj_poses)

        actions = self.agent.act(states=in_states)

        if self.explore > np.random.uniform():
            actions = np.random.uniform(low=0.0, high=1, size=self.num_actions)

        a_in = self.scaleActions(actions)
        self.gripper_open = a_in[-1]>0.5

        self.ee_pos = [self.target_state[0], self.target_state[1], a_in[0]]
        actions2 = list(self.ee_pos) + self.calculateQuaternion(a_in[1]) + list([self.gripper_open])


        self.dist_before_action = max(0.05,np.linalg.norm(self.target_state[:3] - gripper_pose[:3]))
        return actions2


    def scaleActions(self, actions):
   
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
            reward += 50
            if self.ee_pos[-1] > self.target_start_pose[2] + 0.05:
                reward += 700
                terminal = True
                return reward, terminal

        if self.target_state[2] < 0.75 + (self.target_start_pose[2] - 0.75)/2:
            reward -= 10

        if not self.gripper_open and not self.has_object:
            reward -=5

        return reward, terminal



