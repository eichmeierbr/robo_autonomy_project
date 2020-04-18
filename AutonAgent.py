import numpy as np
import scipy as sp
import quaternion as quater
from scipy.spatial.transform import Rotation as R
import copy



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
        self.height_above = 0.1
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
        # if self.reached_goal:
            # t_pos     = goal_loc[:3]
            # t_pos[2] += 0.05
        # else:
        t_pos     = goal_loc[:3]
        t_pos[2] += self.height_above



        des_pos = list(t_pos)


        ######## Orientation Control ##########
        # Robot: X, Y, Z
        # Box: Y, X, -Z
        # target_qt  = obs.gripper_pose[3:]
        # target_qt = goal_loc[3:]
        des_quat = [0, 1,0,0]

        # des_quat = self.orientationControl(target_qt)

        des_quat = list(des_quat)


        ####### Gripper Control ############
        if self.on_target: gripper_pos = [0]
        else: gripper_pos = [1] ### Open: 1, Closed: 0



        # if self.on_target:
        des_pos = list([ 0.228, -0.121, 1.406])
        # else:
        #     des_pos = list([ 0.2, 0, 0.85])
            
        if np.linalg.norm(des_pos[:3] - obs.gripper_pose[:3]) < .01:
            self.on_target = True

            
        return des_pos + des_quat + gripper_pos


    def move_above_object(self, obj_poses, key):
        goal_loc = obj_poses[key]
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        t_pos[2] += self.height_above
        t_pos[2] = 0.99
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = [0, 1,0,0]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        gripper_pos = [1] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos       

        
    def move_to_pos(self, goal_pose):
        ####### Position Control #############
        t_pos    = goal_pose[:3]
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = [0, 1,0,0]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        gripper_pos = [1] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos   


    def orientationControl(self, target):
        r = quater.from_float_array(target)
        r_vect = quater.as_rotation_vector(r)

        cop_vect = copy.copy(r_vect)
        r_vect[0] = cop_vect[1]
        r_vect[1] = cop_vect[0]
        r_vect[2] *=-1


        r = quater.from_rotation_vector(r_vect)
        out = quater.as_float_array(r)
        return out
