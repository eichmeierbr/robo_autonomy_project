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
        self.on_target = False
        self.has_object = False

    def act(self, obs, obj_poses):
        self.obs = obs
        self.obj_poses = obj_poses
        cracker_loc = self.obj_poses['sugar']
        goal_loc = cracker_loc.copy()

        if np.linalg.norm(goal_loc[:3] - obs.gripper_pose[:3]) < self.height_above + .01:
            self.reached_goal = True
        else:
            self.reached_goal = False
            
        ####### Position Control #############
        if self.reached_goal:
            t_pos     = goal_loc[:3]
            t_pos[2] += 0.05
        else:
            t_pos     = goal_loc[:3]
            t_pos[2] += self.height_above

        if np.linalg.norm(t_pos - obs.gripper_pose[:3]) < .01:
            self.on_target = True

        
        delta_pos = self.get_delta_pos(t_pos)


        ######## Orientation Control ##########
        # t_qt  = [0,0.70711,0,70711]
        t_qt  = obs.gripper_pose[3:]
        # t_qt = goal_loc[3:]
        delta_quat = self.get_delta_quat(t_qt)


        ####### Gripper Control ############
        if self.on_target: gripper_pos = [0]
        else: gripper_pos = [1] ### Open: 1, Closed: 0

        return delta_pos + delta_quat + gripper_pos


    def get_delta_pos(self, target):
        ############ Position Control ########
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

        return delta_quat  



class AutonAgentAbsolute_Mode:

    def __init__(self):
        self.reached_goal = False
        self.step_size = 0.01
        self.quat_step = 0.05
        self.height_above = 0.2
        self.obs = []
        self.obj_poses = []
        self.on_target = False
        self.has_object = False

    def act(self, obs, obj_poses):
        self.obs = obs
        self.obj_poses = obj_poses
        cracker_loc = self.obj_poses['sugar']
        goal_loc = cracker_loc.copy()

        if np.linalg.norm(goal_loc[:3] - obs.gripper_pose[:3]) < self.height_above + .01:
            self.reached_goal = True
        else:
            self.reached_goal = False
            
        ####### Position Control #############
        if self.reached_goal:
            t_pos     = goal_loc[:3]
            t_pos[2] += 0.05
        else:
            t_pos     = goal_loc[:3]
            t_pos[2] += self.height_above

        if np.linalg.norm(t_pos - obs.gripper_pose[:3]) < .01:
            self.on_target = True

        des_pos = list(t_pos)


        ######## Orientation Control ##########
        t_qt  = obs.gripper_pose[3:]
        # t_qt = goal_loc[3:]
        des_quat = list(t_qt)


        ####### Gripper Control ############
        if self.on_target: gripper_pos = [0]
        else: gripper_pos = [1] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos
