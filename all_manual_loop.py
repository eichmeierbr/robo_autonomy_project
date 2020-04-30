import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R
import copy
from utils import *

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutGroceriesInCupboard
from pyrep.objects.object import Object
from pyrep.errors import ConfigurationPathError
from AutonAgent import *



##################################################################################
########### Initialize Environment and Agents Includes ###########################

action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
env = Environment(action_mode, '', ObservationConfig(), False)
task = env.get_task(PutGroceriesInCupboard)


manual_agent = AutonAgentAbsolute_Mode()

obj_pose_sensor = NoisyObjectPoseSensor(env)
obj_poses = obj_pose_sensor.get_poses()


descriptions, obs = task.reset()
print(descriptions)

targets = ['crackers_grasp_point', 'mustard_grasp_point', 'coffee_grasp_point', 'sugar_grasp_point','spam_grasp_point', 
            'tuna_grasp_point', 'soup_grasp_point', 'strawberry_jello_grasp_point', 'chocolate_jello_grasp_point']

targets_grab = ['crackers', 'mustard', 'coffee', 'sugar','spam', 
            'tuna', 'soup', 'strawberry_jello', 'chocolate_jello']


########### Initialize Environment and Agents Includes ###########################
##################################################################################


def convertTargetCoordsToWorld(obj_poses, target_coord, target_name='waypoint4'):
    t_pos = obj_poses[target_name]

    # Create Rotation Matrix
    rot_val = R.from_quat(obj_poses[target_name][3:7])
    to_world_mat = rot_val.as_matrix()

    # Make Rotation Matrix an Affine 4x4 matrix
    to_world_mat = np.hstack((to_world_mat,np.array(t_pos[:3]).reshape([-1,1])))
    to_world_mat = np.vstack((to_world_mat, [0,0,0,1]))

    # Convert the Coordinates
    go_pt = np.hstack((target_coord, 1))
    world_pt = to_world_mat @ go_pt.T

    return world_pt[:3]

# def stageGripperAboveTarget():
#     ## Stage point to avoid cupboard
#     actions = manual_agent.move_to_pos([0.25, 0, 0.99])
#     obs, reward, terminal = task.step(actions)

#     ## Stage above object
#     actions = manual_agent.move_above_object(obj_poses, target_name)
#     obs, reward, terminal = task.step(actions)
#     return obs

# def stageGraspedObject(obs):
#     for i in range(3):
#         # Go to pre-stage location
#         actions = [0.25, 0, 0.99,0,1,0,0,0.0]
#         obj_poses = obj_pose_sensor.get_poses()
#         actions[:2] = obs.gripper_pose[:2]
#         obs, reward, terminal = task.step(actions)

#     for i in range(3):
#         # Go to stage position
#         obj_poses = obj_pose_sensor.get_poses()
#         actions = list(obj_poses['waypoint3'])
#         actions.append(0.0)
#         obs, reward, terminal = task.step(actions)
#     return obs

def prepInFrontCupboard(obj_poses, target_coord=np.array([0, 0, -0.05])):
    action= list(obj_poses['waypoint4'])
    action[:3] = list(convertTargetCoordsToWorld(obj_poses, target_coord))
    action.append(0)

    return action


def check_if_in_cupboard_manual(obj_name,obj_poses):
    return (obj_poses[obj_name][2]>1)



def sample_reset_pos(area: Object):
    minx = 0.0
    maxx = 0.45
    miny =  -0.35
    maxy = 0.35

    x = np.random.uniform(minx, maxx)
    y = np.random.uniform(miny, maxy)
    z = 0.95
    print('Reset Location: X=%.2f, Y=%.2f' %(x,y))
    return x, y, z

def resetTask(task):
    obs = task.get_observation()
    obj_poses = obj_pose_sensor.get_poses()
    surface = Object.get_object('worksurface')

    #drop anything in hand55
    actions = manual_agent.ungrasp_object(obs)
    obs, reward, terminal = task.step(actions)

    #get items in cupboard

    in_cupboard = []
    print('started reseting')
    for k in ['crackers', 'mustard', 'coffee', 'sugar','spam', 'tuna', 'soup', 'strawberry_jello', 'chocolate_jello']:
        if check_if_in_cupboard_manual(k,obj_poses):
            in_cupboard.append(k)
    


    #move to start position
    actions = manual_agent.move_to_pos([0.25, 0, 1], False)
    obs, reward, terminal = task.step(actions)
    print('moved to start')

    while len(in_cupboard)>0:
        #move to above object location
        while len(task._robot.gripper._grasped_objects) == 0:
            if(len(in_cupboard)>1):
                random = np.random.randint(len(in_cupboard)-1)
                print(random)
                obj = in_cupboard[random]
            else:
                obj = in_cupboard[0]
            # actions = manual_agent.move_above_cabinet(obj_poses, obj, False)
            actions = prepInFrontCupboard(obj_poses)
            obs, reward, terminal = task.step(actions)
            print('move above cabinet')
            target_obj = Object.get_object(obj)
            #attempt straight grasp
            grasped = False
            actions = manual_agent.move_to_cabinet_object(obj_poses, obj, False)
            prev_forces = obs.joint_forces
            while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01 and not grasped and np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50:
                prev_forces = obs.joint_forces
                print('stepping to target')
                obs, reward, terminate = task.step(actions)

                grasped = task._robot.gripper.grasp(target_obj)
                print(obj, grasped)
            

            #remove from cabinet
            actions = manual_agent.move_above_cabinet_num(obs, obj_poses, 5)
            obs, reward, terminal = task.step(actions)
            print('moved above cabinet_num')
        

        #place on table
        print ('place on table')
        # Go to post-grasp location
        
        actions = [0.25, 0, 0.99, 0, 1, 0, 0, 0]
        prev_forces = obs.joint_forces
        while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01 and (np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50):
            prev_forces = obs.joint_forces
            print('stepping to post-grasp staging')
            obs, reward, terminate = task.step(actions)
        print('moved to post-grasp location', actions, obs.gripper_pose)

        while grasped:
            reset_x, reset_y, reset_z = sample_reset_pos(surface)

            print('Randomly chosen reset location: ', reset_x, ', ', reset_y)

            _, _, _, _, target_zmin, target_zmax = target_obj.get_bounding_box()
            actions = [reset_x, reset_y, reset_z, 0, 1, 0, 0, 0]
            print('Reset location actions: ', actions)
            try:
                prev_forces = obs.joint_forces
                while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01 and (np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50):
                    prev_forces = obs.joint_forces
                    print('stepping to reset location')
                    obs, reward, terminate = task.step(actions)
            except ConfigurationPathError:
                print('Bad choice! Pick again.')
                continue
            print('moved to reset location', actions, obs.gripper_pose)
            task._robot.gripper.release()
            grasped = False
        
        print('nextobject')
        in_cupboard.clear()
        obj_poses = obj_pose_sensor.get_poses()
        for k in ['crackers', 'mustard', 'coffee', 'sugar','spam', 'tuna', 'soup', 'strawberry_jello', 'chocolate_jello']:
            if check_if_in_cupboard_manual(k,obj_poses):
                in_cupboard.append(k)
        
    #open hand
    actions = manual_agent.ungrasp_object(obs)
    obs, reward, terminal = task.step(actions)


    #move to start position
    actions = manual_agent.move_to_pos([0.25, 0, 1])
    obs, reward, terminal = task.step(actions)
    print('finished resetting')
    
    
    descriptions=None
    #original reset task
    #descriptions, obs = task.reset()
    return descriptions, obs


def placeObject(location, obj_pose_sensor):
    obj_poses = obj_pose_sensor.get_poses()
    action = list(obj_poses['waypoint4'])
    pos = convertTargetCoordsToWorld(obj_poses, location)
    action[:3] = pos
    action.append(0)
    obs, reward, terminal = task.step(action)
    action[-1] = 1
    obs, reward, terminal = task.step(action)
    action = list(obj_poses['waypoint3'])
    action.append(1)
    obs, reward, terminal = task.step(action)
    return obs




while True:

    # Initialize Target Params
    obj_poses = obj_pose_sensor.get_poses()
    # target_num =  np.random.randint(0,len(targets)-1)
    # target_name = targets[target_num]
    # target_state = list(obj_poses[target_name])

    targets = ['sugar_grasp_point','mustard_grasp_point', 'soup_grasp_point']
    positions = [np.array([0, 0.075, 0]), np.array([0, -0.11, 0]), np.array([0, -0.03, 0])]
    # targets = ['sugar_grasp_point']
    # positions = [np.array([0, 0.075, 0])]
    
    for target, pos in zip(targets, positions):
        ######### Grasp Object Object #########
        obs = manual_agent.pickup_and_stage_object(target,task,obj_pose_sensor)
        ######### Place Grasped Object #########
        # pos = np.array([0, 0.45, 0])
        obs = placeObject(pos, obj_pose_sensor)
    resetTask(task)


# rl_place_agent.agent.close()
env.shutdown()


