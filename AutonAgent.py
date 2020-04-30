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
        self.obj_poses = []
        self.on_target = False
        self.has_object = False

    def act(self, obs, obj_poses, key):
        self.obs = obs
        self.obj_poses = obj_poses
        cracker_loc = self.obj_poses[key]
        goal_loc = cracker_loc.copy()
        print(obs.gripper_pose)
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
        des_quat = list(self.obj_poses['waypoint3'][3:])

        # des_quat = self.orientationControl(target_qt)

        des_quat = list(des_quat)


        ####### Gripper Control ############
        if self.on_target: gripper_pos = [0]
        else: gripper_pos = [1] ### Open: 1, Closed: 0



        # if self.on_target:
        des_pos = list(self.obj_poses['waypoint3'][:3])
        # else:
        #     des_pos = list([ 0.2, 0, 0.85])
            
        if np.linalg.norm(des_pos[:3] - obs.gripper_pose[:3]) < .01:
            self.on_target = True

            
        return des_pos + des_quat + gripper_pos


    def move_above_object(self, obj_poses, key, grip_open=True):
        goal_loc = obj_poses[key].copy()
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        t_pos[2] = goal_loc[2]+self.height_above
        # t_pos[2] = 0.99
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = [0, 1,0,0]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        if grip_open:
            gripper_pos = [1] ### Open: 1, Closed: 0
        else:
            gripper_pos = [0]

        return des_pos + des_quat + gripper_pos   

    def move_above_object_dep(self, obj_poses, key, depth):
        goal_loc = obj_poses[key].copy()
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        t_pos[2] = goal_loc[2]+self.height_above-depth
        # t_pos[2] = 0.99
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = [0, 1,0,0]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        if True:
            gripper_pos = [1] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos    
    
    def grasp_object(self, obs):
        self.obs = obs
        goal_loc = self.obs.gripper_pose.copy()
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = goal_loc[3:]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        gripper_pos = [0] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos 

    def ungrasp_object(self, obs):
        self.obs = obs
        goal_loc = obs.gripper_pose.copy()
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        t_pos[2] += self.height_above

        # t_pos[2] = 1
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = goal_loc[3:]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        gripper_pos = [1] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos

        
    def move_to_pos(self, goal_pose, grip_open=True):
        ####### Position Control #############
        t_pos    = goal_pose[:3]
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = [0, 1,0,0]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        if grip_open:
            gripper_pos = [1] ### Open: 1, Closed: 0
        else:
            gripper_pos = [0]

        return des_pos + des_quat + gripper_pos   

    def move_above_cabinet(self, obj_poses, object_name, grip_open=True):
        waypoint = obj_poses['waypoint3'].copy()
        goal_loc = obj_poses[object_name].copy()
        ####### Position Control #############
        t_pos   = goal_loc[:3]                                                                      #v [down-up,left-right,in-out)]
        t_pos   = t_pos + quater.as_rotation_matrix(quater.from_float_array(waypoint[3:]))@np.array([0,0,(goal_loc[2]-waypoint[2])*3])
        t_pos = waypoint[:3]
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = waypoint[3:]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        if grip_open:
            gripper_pos = [1] ### Open: 1, Closed: 0
        else:
            gripper_pos = [0]

        return des_pos + des_quat + gripper_pos  

    def move_above_cabinet_num(self, obs, obj_poses, object_number):
        waypoint = obj_poses['waypoint3'].copy()
        goal_loc = waypoint.copy()
        ####### Position Control #############
        t_pos   = goal_loc[:3]                                                                      #v [down-up,left-right,in-out)]
        t_pos   = t_pos + quater.as_rotation_matrix(quater.from_float_array(goal_loc[3:]))@np.array([0,.1,0])
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = waypoint[3:]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        if obs.gripper_open:
            gripper_pos = [1] ### Open: 1, Closed: 0
        else:
            gripper_pos = [0]

        return des_pos + des_quat + gripper_pos  

    def move_to_cabinet_object(self, obj_poses, object_name, grip_open=True):
        waypoint = obj_poses['waypoint3'].copy()
        goal_loc = obj_poses[object_name].copy()
        ####### Position Control #############
        t_pos   = goal_loc[:3]
        #t_pos   = t_pos + quater.as_rotation_matrix(quater.from_float_array(goal_loc[3:]))@np.array([0,spot,0])
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = waypoint[3:]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        if grip_open:
            gripper_pos = [1] ### Open: 1, Closed: 0
        else:
            gripper_pos = [0]

        return des_pos + des_quat + gripper_pos  

    def move_into_cabinet(self, obs, obj_poses, object_number):
        self.obs = obs
        goal_loc = obj_poses['waypoint4'].copy()
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        spot = -.11+.05*object_number
        if (spot>.12):
            spot = -.11+.05*(object_number-4)
        t_pos   = t_pos + quater.as_rotation_matrix(quater.from_float_array(goal_loc[3:]))@np.array([-spot/1.5,spot,.01])
        des_pos = list(t_pos)
        ######## Orientation Control ##########
        des_quat = goal_loc[3:]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        gripper_pos = [self.obs.gripper_open] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos 

    def align_gripper(self, obj_poses, key):
        goal_loc = obj_poses[key]
        ####### Position Control #############
        t_pos    = goal_loc[:3]
        t_pos[2] += self.height_above
        t_pos[2] = 0.99
        des_pos = list(t_pos)

        ######## Orientation Control ##########
        des_quat = [0, 1,0,0]
        des_quat[:2] = goal_loc[3:5]
        des_quat = list(des_quat)

        ####### Gripper Control ############
        gripper_pos = [1] ### Open: 1, Closed: 0

        return des_pos + des_quat + gripper_pos  


    def pickup_and_stage_object(self, target_name, task, obj_pose_sensor):
        obj_poses = obj_pose_sensor.get_poses()
        ## Stage point to avoid cupboard
        actions = [0.25, 0, 0.99,0,1,0,0,1]
        obs, reward, terminal = task.step(actions)

        ## Stage above object
        obj_poses = obj_pose_sensor.get_poses()

        actions = self.move_above_object(obj_poses, target_name)
        actions[3:7] = obj_poses[target_name][3:7]
        obs, reward, terminal = task.step(actions)

        ## Drop Down To Object
        obj_poses = obj_pose_sensor.get_poses()
        des_pos = list(obj_poses[target_name])
        actions = self.move_to_pos(des_pos)
        actions[3:7] = obs.gripper_pose[3:7]
        obs, reward, terminal = task.step(actions)

        ## Grasp Object
        obj_poses = obj_pose_sensor.get_poses()
        actions = list(obs.gripper_pose)
        actions.append(0)
        obs, reward, terminal = task.step(actions)

        # Go to pre-stage location
        obj_poses = obj_pose_sensor.get_poses()
        actions = [0.25, 0, 0.99,0,1,0,0,0]
        actions[:2] = obs.gripper_pose[:2]
        obs, reward, terminal = task.step(actions)


        # Go to pre-stage location
        obj_poses = obj_pose_sensor.get_poses()
        actions = [0.25, 0, 0.99,0,1,0,0,0]
        actions[:2] = obs.gripper_pose[:2]
        obs, reward, terminal = task.step(actions)

        # Go to stage position
        obj_poses = obj_pose_sensor.get_poses()
        actions = list(obj_poses['waypoint3'])
        actions.append(0)
        obs, reward, terminal = task.step(actions)

        # Go to stage position
        obj_poses = obj_pose_sensor.get_poses()
        actions = list(obj_poses['waypoint3'])
        actions.append(0)
        obs, reward, terminal = task.step(actions)
        return obs
    


    def graspObjectOnceAbove(self, target_name, task, obj_pose_sensor):
        ## Stage above object
        obj_poses = obj_pose_sensor.get_poses()

        actions = self.move_above_object(obj_poses, target_name)
        actions[3:7] = obj_poses[target_name][3:7]
        obs, reward, terminal = task.step(actions)

        ## Drop Down To Object
        obj_poses = obj_pose_sensor.get_poses()
        des_pos = list(obj_poses[target_name])
        actions = self.move_to_pos(des_pos)
        actions[3:7] = obs.gripper_pose[3:7]
        obs, reward, terminal = task.step(actions)

        ## Grasp Object
        obj_poses = obj_pose_sensor.get_poses()
        actions = list(obs.gripper_pose)
        actions.append(0)
        obs, reward, terminal = task.step(actions)
        return obs, True


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
