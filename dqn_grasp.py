import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import copy

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutGroceriesInCupboard

from AutonAgent import *



############################################################
########### TensorForce Includes ###########################
import sys
sys.path.append('../')
sys.path.append(sys.path[0] + '/TensorForceFiles')

import gym
import rlbench.gym
from tensorforce import Agent

from DQN_class import *
from dqn_grasp_class import *
from dqn_grasp_class_2 import *
from TensorForce_class import *
########### TensorForce Includes ###########################
############################################################


def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01]  * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            # pose[:3] += pos
            # pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose
        return obj_poses



if __name__ == "__main__":
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes

    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    len_episode = 10
    save_name = 'dqn_grasp_2'

    RLagent.len_episode = len_episode
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    
    descriptions, obs = task.reset()
    print(descriptions)

    agent2 = AutonAgentAbsolute_Mode(obs)
    RLagent = DQN_grasp_class_2(load = save_name)
    # agent = TensorForceClass(load='dqn_grasp')

    targets = ['crackers_grasp_point', 'mustard_grasp_point', 'coffee_grasp_point', 'sugar_grasp_point','spam_grasp_point', 
                'tuna_grasp_point', 'soup_grasp_point', 'strawberry_jello_grasp_point', 'chocolate_jello_grasp_point']
    episode_num =0
    rews = []
    save_freq = 10

    while True:
        episode_num += 1
        total_reward = 0
        obj_poses = obj_pose_sensor.get_poses()
        RLagent.target_num = np.random.randint(0,len(targets)-1)
        target_name = targets[RLagent.target_num]
        target_state = list(obj_poses[target_name])
        RLagent.target_start_pose = copy.deepcopy(target_state)
        RLagent.target_state = target_state


        try:
            ## Stage point to avoid cupboard
            actions = agent2.move_to_pos(obs, [0.25, 0, 0.99])
            obs, reward, terminal = task.step(actions)
    
            ## Stage above object
            actions = agent2.move_above_object(obj_poses, target_name, 0)
            obs, reward, terminal = task.step(actions)
        except:
            descriptions, obs = task.reset()
            continue

        best_reward = 0

        for i in range(len_episode):
            # Getting noisy object poses
            obj_poses = obj_pose_sensor.get_poses()
    
            # Getting various fields from obs
            current_joints = obs.joint_positions
            gripper_pose = obs.gripper_pose
            rgb = obs.wrist_rgb
            depth = obs.wrist_depth
            mask = obs.wrist_mask

            # Update arm states
            RLagent.has_object = len(task._robot.gripper._grasped_objects) > 0

            actions = RLagent.act(obs,obj_poses, key=target_name)

            try:
                obs, reward, terminal = task.step(actions)
        
                ### Check current distance
                RLagent.dist_after_action = max(0.05,np.linalg.norm(target_state[:3] - obs.gripper_pose[:3]))
        
                ### Calculate reward
                reward, terminal = RLagent.calculateReward()

            except:
                reward = -RLagent.dist_before_action*5
                terminal = False

            ## Observe results
            RLagent.agent.observe(terminal=terminal, reward=reward)
            total_reward += reward

            if terminal: 
                break

            print('Iteration: %i, Reward: %.1f' %(i, reward))

        print('Episode: %i, Total Reward: %.1f' %(episode_num,total_reward))
        rews.append(total_reward)
        print('Reset')
        if episode_num % save_freq == save_freq - 1: RLagent.agent.save(directory=save_name)
        descriptions, obs = task.reset()
        RLagent.agent.reset()


        

    env.shutdown()
    RLagent.agent.close()


