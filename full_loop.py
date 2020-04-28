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



############################################################
########### TensorForce Includes ###########################
import sys
sys.path.append('../')
sys.path.append(sys.path[0] + '/TensorForceFiles')

import gym
import rlbench.gym
from tensorforce import Agent

from dqn_place_class import *
from dqn_grasp_class_2 import *
########### TensorForce Includes ###########################
############################################################


##################################################################################
########### Initialize Environment and Agents Includes ###########################

action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
env = Environment(action_mode, '', ObservationConfig(), False)
task = env.get_task(PutGroceriesInCupboard)

len_episode = 10
place_name = 'dqn_place'
grasp_name = 'dqn_grasp_2'

manual_agent = AutonAgentAbsolute_Mode()
rl_place_agent = DQN_place(load = place_name)
rl_grasp_agent = DQN_grasp_class_2(load = grasp_name)
rl_place_agent.len_episode = len_episode
rl_grasp_agent.len_episode = len_episode

obj_pose_sensor = NoisyObjectPoseSensor(env)
obj_poses = obj_pose_sensor.get_poses()


descriptions, obs = task.reset()
print(descriptions)

targets = ['crackers_grasp_point', 'mustard_grasp_point', 'coffee_grasp_point', 'sugar_grasp_point','spam_grasp_point', 
            'tuna_grasp_point', 'soup_grasp_point', 'strawberry_jello_grasp_point', 'chocolate_jello_grasp_point']
grasp_episode_num = 1
place_episode_num = 1
grasp_success = 0
place_success = 0
save_freq = 10

########### Initialize Environment and Agents Includes ###########################
##################################################################################

def stageGripperAboveTarget():
    ## Stage point to avoid cupboard
    actions = manual_agent.move_to_pos([0.25, 0, 0.99])
    obs, reward, terminal = task.step(actions)

    ## Stage above object
    actions = manual_agent.move_above_object(obj_poses, target_name)
    obs, reward, terminal = task.step(actions)
    return obs


def rlGraspObject(RLagent, obs):
    global grasp_episode_num
    grasp_episode_num += 1
    total_reward = 0
    success = False

    for i in range(len_episode):
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()

        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose

        # Update arm states
        RLagent.has_object = len(task._robot.gripper._grasped_objects) > 0

        actions = RLagent.act(obs,obj_poses, key=target_name)

        try:
            obs, reward, terminal = task.step(actions)
    

            ### Check current distance
            target_state = list(obj_poses[target_name])
            RLagent.dist_after_action = max(0.05,np.linalg.norm(target_state[:3] - obs.gripper_pose[:3]))
    
            ### Calculate reward
            RLagent.has_object = len(task._robot.gripper._grasped_objects) > 0
            reward, terminal = RLagent.calculateReward()

        except:
            reward = -RLagent.dist_before_action*5
            success = False

        ## Observe results
        RLagent.agent.observe(terminal=terminal, reward=reward)
        total_reward += reward

        if RLagent.has_object:
            success = True
            break

        # print('Iteration: %i, Reward: %.1f' %(i, reward))

    # print('Grasp Episode: %i, Total Reward: %.1f' %(grasp_episode_num,total_reward))
    if grasp_episode_num % save_freq == save_freq - 1: RLagent.agent.save(directory=grasp_name)
    RLagent.agent.reset()
    return obs, success


def stageGraspedObject(obs):
    # while np.linalg.norm(obs.gripper_pose[:3] - [0.25, 0, 0.99] > 0.1):
    for i in range(3):
        # Go to pre-stage location
        actions = [0.25, 0, 0.99,0,1,0,0,0]
        obj_poses = obj_pose_sensor.get_poses()
        actions[:2] = obs.gripper_pose[:2]
        obs, reward, terminal = task.step(actions)

    while np.linalg.norm(obs.gripper_pose[:3] - obj_poses['waypoint3'][:3] > 0.01):
        # Go to stage position
        obj_poses = obj_pose_sensor.get_poses()
        actions = list(obj_poses['waypoint3'])
        actions.append(0)
        obs, reward, terminal = task.step(actions)
    return obs


def rlPlaceObject(rl_place_agent, obs):
    total_reward = 0
    global place_episode_num
    place_episode_num += 1
    for i in range(len_episode):
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()

        actions = rl_place_agent.act(obs,obj_poses, key=target_name)

        try:
            obs, reward, terminal = task.step(actions)
            
            ### Check current distance
            rl_place_agent.dist_after_action = max(0.05,np.linalg.norm(target_state[:3] - obs.gripper_pose[:3]))

            # Update arm states
            rl_place_agent.has_object = len(task._robot.gripper._grasped_objects) > 0
    
            obj_poses = obj_pose_sensor.get_poses()
            rl_place_agent.gripper_pose = obs.gripper_pose

            rl_place_agent.obj_poses = obj_poses
            ### Calculate reward
            reward, terminal = rl_place_agent.calculateReward()

        except:
            reward = 0
            terminal = False

        if terminal:
            # Step back to staging point
            obj_poses = obj_pose_sensor.get_poses()
            actions = list(obj_poses['waypoint3'])
            actions.append(obs.gripper_open*1)
            obs, reward, terminal = task.step(actions)

            # Check where the target dropped
            obj_poses = obj_pose_sensor.get_poses()
            rl_place_agent.obj_poses = obj_poses
            rl_place_agent.gripper_pose = obs.gripper_pose

            # Update Reward
            reward, terminal = rl_place_agent.calculateReward()
            rl_place_agent.agent.observe(terminal=terminal, reward=reward)
            total_reward += reward
            if reward == -1: terminal = False
            # print('Episode: %i, Total Reward: %.1f' %(place_episode_num,total_reward))
            break

        ## Observe results
        rl_place_agent.agent.observe(terminal=terminal, reward=reward)
        total_reward += reward

        # print('Iteration: %i, Reward: %.1f' %(i, reward))
    
    # print('Place Episode: %i, Total Reward: %.1f' %(place_episode_num,total_reward))
    if place_episode_num % save_freq == save_freq - 1: rl_place_agent.agent.save(directory=place_name)
    rl_place_agent.agent.reset()

    # Try to drop the object and see what happens
    actions = list(obs.gripper_pose) + ([1])
    obs, reward, terminal = task.step(actions)

    actions = list(obj_poses['waypoint3'])
    actions.append(1)
    obs, reward, terminal = task.step(actions)
    obj_poses = obj_pose_sensor.get_poses()
    rl_place_agent.obj_poses = obj_poses
    success = rl_place_agent.is_in_cupboard()
    return obs, success


def check_if_in_cupboard(target_name, obj_poses):
    t_pos = obj_poses[target_name[:-12]]
    t_pos = rl_place_agent.convertTargetCoordsToCabinet(t_pos)
    in_cab =            t_pos[0] <= rl_place_agent.x_r[1] + 0.05
    # in_cab = in_cab and rl_place_agent.y_r[0] - 0.05 <= t_pos[1] <= rl_place_agent.y_r[1] + 0.05
    in_cab = in_cab and rl_place_agent.z_r[0] - 0.05 <= t_pos[2]
    return in_cab

def sample_reset_pos(area: Object):
    minx, maxx, miny, maxy, _, _ = area.get_bounding_box()
    pose = area.get_pose()
    print('Surface pose: ', pose)

    x = np.random.uniform(minx, maxx) + pose[0]
    y = np.random.uniform(miny, maxy) + pose[1]
    z = pose[2] + 0.05

    return x, y, z

def resetTask(task):
    obs = task.get_observation()
    obj_poses = obj_pose_sensor.get_poses()
    rl_place_agent.obj_poses = obj_poses
    surface = Object.get_object('worksurface')

    #drop anything in hand
    while not obs.gripper_open:
        actions = list(obs.gripper_pose) + [1]
        obs, reward, terminal = task.step(actions)

    #get items in cupboard
    in_cupboard = []
    print('started reseting')
    for k in targets:
        if check_if_in_cupboard(k,obj_poses):
            in_cupboard.append(k)

    #move to start position
    actions = manual_agent.move_to_pos([0.25, 0, 1])
    obs, reward, terminal = task.step(actions)
    print('moved to start')

    for obj in in_cupboard:
        #move to above object location
        actions = manual_agent.move_above_cabinet(obj_poses, obj)
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
        
        #if failed kick the object to the back of the line and try another
        if (not grasped):
            in_cupboard.append(obj)
            print('kicking')
            continue

        #remove from cabinet
        actions = manual_agent.move_above_cabinet_num(obs, obj_poses, 5)
        obs, reward, terminal = task.step(actions)
        print('moved above cabinet_num')

        #place on table
        print ('place on table')
        # Go to post-grasp location
        actions = [0.25, 0, 0.99, 0, 1, 0, 0, 0]
        while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01:
            print('stepping to post-grasp staging')
            obs, reward, terminate = task.step(actions)
        print('moved to post-grasp location', actions, obs.gripper_pose)

        while grasped:
            reset_x, reset_y, reset_z = sample_reset_pos(surface)

            print('Randomly chosen reset location: ', reset_x, ', ', reset_y)

            _, _, _, _, target_zmin, target_zmax = target_obj.get_bounding_box()
            actions = [reset_x, reset_y, target_zmax - target_zmin + reset_z, 0, 1, 0, 0, 0]
            print('Reset location actions: ', actions)
            try:
                while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01:
                    print('stepping to reset location')
                    obs, reward, terminate = task.step(actions)
            except ConfigurationPathError:
                print('Bad choice! Pick again.')
                continue
            print('moved to reset location', actions, obs.gripper_pose)
            task._robot.gripper.release()
            grasped = False
        
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

while True:
    ## Initialize Episode ##
    # Print Success Info
    print('Grasp Success: %.2f' %(grasp_success/grasp_episode_num))
    print('Place Success: %.2f' %(place_success/place_episode_num))
    print('Total Success: %.2f' %(place_success/grasp_episode_num))
    # Initialize Episode Params
    total_reward = 0
    # Initialize Target Params
    obj_poses = obj_pose_sensor.get_poses()
    target_num =  np.random.randint(0,len(targets)-1)
    target_name = targets[target_num]
    target_state = list(obj_poses[target_name])
    # Set target info in RL classes
    rl_place_agent.target_num = target_num
    rl_place_agent.target_name = target_name
    rl_grasp_agent.target_num = target_num
    rl_grasp_agent.target_name = target_name
    ## End Initialize Episode

    ######### Stage Gripper above Target #############
    try:
        print('Prepare Gripper for RL Grasp')
        obs = stageGripperAboveTarget()
    except:
        descriptions, obs = resetTask(task)
        continue
    ######### END Stage Gripper above Target ##########

    ######### Grasp Object #########
    try:
        print('Grasping %s with RL Agent' %(target_name[:-12]))
        obj_poses = obj_pose_sensor.get_poses()
        obs, success = rlGraspObject(rl_grasp_agent, obs) #### RL Grasp Object #####
        # while not len(task._robot.gripper._grasped_objects) > 0:
            # obs, success = manual_agent.graspObjectOnceAbove(target_name, task,obj_pose_sensor) #### Manual Grasp Object  #####


        if not success:
            print('RL Grasp Failed')
            descriptions, obs = resetTask(task)
            continue
        else:
            print('RL Agent Successfully Grapsed %s' %(target_name[:-12]))
            manual_agent.has_object = True
            rl_grasp_agent.has_object = True
            rl_place_agent.has_object = True
            grasp_success += 1

    except:
        descriptions, obs = resetTask(task)
        continue
    ######### End Grasp Object #########

    ######### Stage Grasped Object #########
    try:
        print('Staging %s for Placement' %(target_name[:-12]))
        obj_poses = obj_pose_sensor.get_poses()
        obs = stageGraspedObject(obs)
    except:
        descriptions, obs = resetTask(task)
        continue
    ######### End Stage Grasped Object #########

    ######### Place Grasped Object #########
    try:
        print('Placing %s with RL Agent' %(target_name[:12]))
        obj_poses = obj_pose_sensor.get_poses()
        obs, success = rlPlaceObject(rl_place_agent, obs) #### RL Grasp Object #####

        ## Need to finish end conditions
        if not success:
            print('%s Placement Failed' %(target_name[:-12]))
            descriptions, obs = resetTask(task)
            continue
        place_success += 1
        print('Woohoo! You successfully placed ' +  target_name[:-12] + ' in the cupboard!!!')
    except:
        descriptions, obs = resetTask(task)
        continue
    ######### Place Grasped Object #########

    

# rl_place_agent.agent.close()
env.shutdown()


