import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R



class AutonAgentDelta_Mode:

    def __init__(self):
        self.reached_goal = False
        self.step_size = 0.01
        self.quat_step = 0.05
        self.height_above = 0.2
        self.obs = []
        self.obj_poses = []

    def act(self, obs, obj_poses):
        self.obs = obs
        self.obj_poses = obj_poses
        cracker_loc = self.obj_poses['crackers']
        goal_loc = cracker_loc.copy()

        if np.linalg.norm(goal_loc[:3] - obs.gripper_pose[:3]) < self.height_above + .01:
            self.reached_goal = True
        else:
            self.reached_goal = False
            
        # Position Control
        delta_pos = self.get_delta_pos(goal_loc[:3])


        ######## Orientation Control ##########
        # t_qt  = [0,0.70711,0,70711]
        t_qt  = obs.gripper_pose[3:]
        # t_qt = goal_loc[3:]
        delta_quat = self.get_delta_quat(t_qt)


        ####### Gripper Control ############
        gripper_pos = [1] ### Open: 1, Closed: 0
        return delta_pos + delta_quat + gripper_pos


    def get_delta_pos(self, target):
        ############ Position Control ########
        target[2] += self.height_above
        delta_pos = list(target[:3] - self.obs.gripper_pose[:3])

        dist = np.linalg.norm(delta_pos)
        if dist > self.step_size:
            delta_pos = delta_pos/dist*self.step_size 

        return list(delta_pos)


    def get_delta_quat(self, target_quat):
        my_qt = self.obs.gripper_pose[3:]
        my_rot = R.from_quat(my_qt)

        t_rot = R.from_quat(target_quat)
        delta_rot = my_rot * t_rot.inv()
    
        delta_quat = list(delta_rot.as_quat())

        # my_quat = quaternion(my_qt[0], my_qt[1], my_qt[2], my_qt[3])
        # t_quat  = quaternion(t_qt[0], t_qt[1], t_qt[2], t_qt[3])
        # delta_quat = my_quat - t_quat

        # if abs(delta_quat.w) > self.quat_step:
        #     delta_quat.w = self.quat_step * np.sign(delta_quat.w)

        
        # delta_quat = list(delta_quat.as_float_array())

        return delta_quat  


class AutonAgentAbsolute_Mode:

    def __init__(self):
        self.reached_goal = False
        self.step_size = 0.01
        self.quat_step = 0.1

    def act(self, obs, obj_poses):
        cracker_loc = obj_poses['crackers']
        goal_loc = cracker_loc.copy()
        height_above = 0.15

        if np.linalg.norm(goal_loc[:3] - obs.gripper_pose[:3]) < .05:
            self.reached_goal = True
        else:
            self.reached_goal = False
            

        ############ Position Control ########
        goal_loc[2] += height_above
        delta_pos = list(obs.gripper_pose[:3] - goal_loc[:3])

        dist = np.linalg.norm(delta_pos)
        if dist > self.step_size:
            delta_pos = delta_pos/dist*self.step_size 

        goal_pos = list(obs.gripper_pose[:3] - delta_pos)


        ######## Orientation Control ##########
        seq = 'xyz'
        my_qt = obs.gripper_pose[3:]
        my_rot = R.from_quat(my_qt)
        my_rpy = my_rot.as_euler(seq)

        t_qt  = [0,0,0,1]
        # t_qt  = obs.gripper_pose[3:]
        t_rot = R.from_quat(t_qt)
        t_rpy = t_rot.as_euler(seq)
        # t_rpy[2] += np.pi/2
        # t_rpy[0] += np.pi/2

        delta_rpy = my_rpy - t_rpy

        dist = np.linalg.norm(delta_rpy)
        if dist > self.quat_step:
            delta_rpy = delta_rpy/dist * self.quat_step 

        delta_rot = R.from_euler(seq,delta_rpy)
        # goal_rpy = delta_pos + my_rpy

        # goal_rot = R.from_euler(seq,goal_rpy)
        goal_rot = my_rot * delta_rot.inv()
        goal_quat = list(goal_rot.as_quat())


        my_quat = quaternion(my_qt[0], my_qt[1], my_qt[2], my_qt[3])
        t_quat  = quaternion(t_qt[0], t_qt[1], t_qt[2], t_qt[3])
        delta_quat = my_quat - t_quat
        if abs(delta_quat.w) > self.quat_step:
            delta_quat.w = self.quat_step * np.sign(delta_quat.w)

        
        goal_quat = list(delta_quat.as_float_array())


        ####### Gripper Control ############
        gripper_pos = [1]
        return goal_pos + goal_quat + gripper_pos

