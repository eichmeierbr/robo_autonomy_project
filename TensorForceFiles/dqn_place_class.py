from tensorforce import Agent
from TensorForceRL_parent import *
from dqn_grasp_class import *
import numpy as np
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R



class DQN_place(DQN_grasp):

    def __init__(self, num_states=16, num_actions=5, load=None):
        self.num_states = 16 # Object pose, cabinet pose, Gripper Open, Target_num
        self.num_actions = 4 # X, Y, Z, yaw, grasp
        super().__init__(num_states=num_states, num_actions=num_actions, load=load)
        
        self.x_r = [-0.1, 0.05] # Cupboard coords: Point down
        self.y_r = [-0.1, 0.1]  # Cupboad coords: Point to left of cupboard
        self.z_r = [-0.275, 0.1] # Cupboard coords: Point to cupboard
        self.z_r = [-0.1, 0.1] # Cupboard coords: Point to cupboard
        self.yaw_r = [0, np.pi] 
        self.explore = 0.5
        self.stage_point = []
        self.obj_poses = []

    def createRLagent(self, load):
        states_dict = {'type': 'float', 'shape': self.num_states}
        actions_dict = {'type': 'float', 'shape': self.num_actions, 'min_value': self.input_low, 'max_value': self.input_high}

        return Agent.create(
            agent='dqn',
            states = states_dict,  # alternatively: states, actions, (max_episode_timesteps)
            actions = actions_dict,
            memory=10000,
            exploration=0.75,
            max_episode_timesteps= self.len_episode,
        )

    def getInStates(self, key, obj_poses):
        target_state = list(obj_poses[key])
        self.target_state = target_state
        self.has_object = False

        in_states = list(self.ee_pos)
        in_states.extend(list(self.target_state))

        in_states.extend(list([self.gripper_open*1.0, self.target_num]))
        return in_states

        

    def act(self, obs, obj_poses, key='crackers'):
        self.stage_point = obj_poses['waypoint3']
        gripper_pose = obs.gripper_pose
        self.ee_pos = gripper_pose
        self.obj_poses = obj_poses

        in_states = self.getInStates('waypoint4', obj_poses)

        actions = self.agent.act(states=in_states)

        # if self.explore > np.random.uniform():
            # actions = np.random.uniform(low=0.0, high=1, size=self.num_actions)

        a_in = self.scaleActions(actions)
        self.gripper_open = a_in[-1]>0.5

        self.ee_pos = a_in[:3]
        # actions2 = list(self.ee_pos) + self.calculateQuaternion(a_in[3]) + list([self.gripper_open])
        actions2 = list(self.ee_pos) + list(self.stage_point[3:7]) + list([self.gripper_open])

        self.dist_before_action = max(0.05,np.linalg.norm(self.target_state[:3] - gripper_pose[:3]))

        return actions2


    def scaleActions(self, actions):
   
        # Scale all the Actions
        x_pre      = actions[0]*(self.x_r[1] -   self.x_r[0])   + self.x_r[0]
        y_pre      = actions[1]*(self.y_r[1] -   self.y_r[0])   + self.y_r[0]
        z_pre      = actions[2]*(self.z_r[1] -   self.z_r[0])   + self.z_r[0]
        actions[3] = actions[3]*(self.yaw_r[1] - self.yaw_r[0]) + self.yaw_r[0]


        trans = np.array([x_pre, y_pre, z_pre])

        # Convert the Cabinet Frame Coordinates into the World Fram Coordinates
        trans = self.convertTargetCoordsToWorld(trans)

        actions[:3] = trans
        if self.has_object: actions[-1] = 0

        return actions
    

    def calculateReward(self):
        if self.gripper_open:
            t_pos = self.obj_poses[self.target_name[:-12]]
            t_pos = self.convertTargetCoordsToCabinet(t_pos)
            in_cab = self.x_r[0] <= t_pos[0] <= self.x_r[1]
            in_cab = in_cab and self.y_r[0] <= t_pos[1] <= self.z_r[1]
            in_cab = in_cab and self.z_r[0] <= t_pos[2] <= self.z_r[1]
            if not in_cab:
                reward = -1
            else:
                reward = 5
            terminal = True

        else:
            terminal = False
            reward = 0

        return reward, terminal

    def calculateQuaternion(self, angle):
        firstElement  = np.sin(angle/2)
        secondElement = -np.cos(angle/2)
        return [0, secondElement, 0, firstElement]

 
    #### CODE TO TRANSLATE BETWEEN CUPBOARD AND WORLD FRAMES IMPLEMENT WITH RL SCALE AND ACTIONS
    def convertTargetCoordsToWorld(self, target_coord, target_name='waypoint4'):
        t_pos = self.obj_poses[target_name]

        # Create Rotation Matrix
        rot_val = R.from_quat(self.obj_poses[target_name][3:7])
        to_world_mat = rot_val.as_matrix()

        # Make Rotation Matrix an Affine 4x4 matrix
        to_world_mat = np.hstack((to_world_mat,np.array(t_pos[:3]).reshape([-1,1])))
        to_world_mat = np.vstack((to_world_mat, [0,0,0,1]))

        # Convert the Coordinates
        go_pt = np.hstack((target_coord, 1))
        world_pt = to_world_mat @ go_pt.T

        return world_pt[:3]

    def convertTargetCoordsToCabinet(self, target_coord, target_name='waypoint4'):
        t_pos = self.obj_poses[target_name]

        # Create Rotation Matrix
        rot_val = R.from_quat(self.obj_poses[target_name][3:7])
        to_world_mat = rot_val.as_matrix()

        # Make Rotation Matrix an Affine 4x4 matrix
        to_world_mat = np.hstack((to_world_mat,np.array(t_pos[:3]).reshape([-1,1])))
        to_world_mat = np.vstack((to_world_mat, [0,0,0,1]))
        to_cupboard_mat = np.linalg.inv(to_world_mat)

        # Convert the Coordinates
        world_pt = np.hstack((target_coord[:3],1))
        cab_pt = to_cupboard_mat @ world_pt.T

        return cab_pt