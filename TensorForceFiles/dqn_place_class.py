from tensorforce import Agent
from TensorForceRL_parent import *
from dqn_grasp_class import *
import numpy as np
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R



class DQN_place(DQN_grasp):

    def __init__(self, num_states=16, num_actions=3, load=None):
        self.num_states = 16 # Object pose, cabinet pose, Gripper Open, Target_num
        self.num_actions = 5 # X, Y, Z, yaw, grasp
        super().__init__(num_states=num_states, num_actions=num_actions, load=load)
        
        self.x_r = [-0.1, 0.05] # Cupboard coords: Point down
        self.y_r = [-0.1, 0.1]  # Cupboad coords: Point to left of cupboard
        self.z_r = [-0.275, 0.1] # Cupboard coords: Point to cupboard
        self.yaw_r = [0, np.pi] 
        self.explore = 0.3
        self.stage_point = []


    def getInStates(self, key, obj_poses):
        target_state = list(obj_poses[key])
        self.target_state = target_state
        self.has_object = False

        in_states = list(self.ee_pos)
        in_states.extend(list(self.target_state))

        in_states.extend(list([self.gripper_open*1.0, self.target_num]))
        return in_states

        

    def act(self, obs, obj_poses, key='waypoint4'):
        self.stage_point = obj_poses['waypoint3']
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
   
        # Scale all the Actions
        x_pre      = actions[0]*(self.x_r[1] -   self.x_r[0])   + self.x_r[0]
        y_pre      = actions[1]*(self.y_r[1] -   self.y_r[0])   + self.y_r[0]
        z_pre      = actions[2]*(self.z_r[1] -   self.z_r[0])   + self.z_r[0]
        actions[3] = actions[3]*(self.z_r[1] -   self.z_r[0])   + self.z_r[0]
        actions[4] = actions[4]*(self.yaw_r[1] - self.yaw_r[0]) + self.yaw_r[0]


        trans = np.array([x_pre, y_pre, z_pre])

        # Convert the Cabinet Frame Coordinates into the World Fram Coordinates
        rot_val = R.from_quat(self.ee_pos[3:7])
        rot_mat = rot_val.as_matrix()
        trans = rot_mat @ trans

        # if self.has_object: actions[-1] = 0

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
            reward += 30
            if self.ee_pos[-1] > self.target_start_pose[2] + 0.05:
                reward += 350
                terminal = True
                return reward, terminal

        if self.target_state[2] < 0.75 + (self.target_start_pose[2] - 0.75)/2:
            reward -= 10

        if not self.gripper_open and not self.has_object:
            reward -=1

        return reward, terminal



 
#### CODE TO TRANSLATE BETWEEN CUPBOARD AND WORLD FRAMES IMPLEMENT WITH RL SCALE AND ACTIONS
####        actions = list(obj_poses['waypoint3']) + [0]
####        obs, reward, terminal = task.step(actions)
####
####        target_name = 'waypoint4'
####        t_pos = obj_poses[target_name]
####        actions = list(obj_poses[target_name]) + [0]
####        obs, reward, terminal = task.step(actions)
####
####
####        rot_val = R.from_quat(obj_poses[target_name][3:7])
####        to_world_mat = rot_val.as_matrix()
####
####        to_world_mat = np.hstack((to_world_mat,np.array(t_pos[:3]).reshape([-1,1])))
####        to_world_mat = np.vstack((to_world_mat, [0,0,0,1]))
####
####        to_cabinet_mat = np.linalg.inv(to_world_mat)
####
####        stage_pt = np.hstack((obj_poses['waypoint3'][:3],1))
####        cab_pt = to_cabinet_mat @ stage_pt.T
####
####        go_pt = np.array([0, 0, 0.1,1])
####        world_pt = to_world_mat @ go_pt.T
####
####        actions = list(world_pt[:3]) + list(obj_poses[target_name][3:7]) + [0]
####        obs, reward, terminal = task.step(actions)
