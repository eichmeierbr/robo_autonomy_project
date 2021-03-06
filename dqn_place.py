import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R
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

from dqn_place_class import *
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
    save_name = 'dqn_place'

    manual_agent = AutonAgentAbsolute_Mode()
    rl_place_agent = DQN_place(load = save_name)
    # rl_place_agent = DQN_place(load = None)
    rl_place_agent.len_episode = len_episode

    obj_pose_sensor = NoisyObjectPoseSensor(env)
    
    descriptions, obs = task.reset()
    print(descriptions)

    manual_agent = AutonAgentAbsolute_Mode()
        # rl_grasp_agent = DQN_place(load = save_name)
        # rl_grasp_agent = DQN_place(load = None)
        # rl_grasp_agent.len_episode = len_episode

    targets = ['crackers_grasp_point', 'mustard_grasp_point', 'coffee_grasp_point', 'sugar_grasp_point','spam_grasp_point', 
                'tuna_grasp_point', 'soup_grasp_point', 'strawberry_jello_grasp_point', 'chocolate_jello_grasp_point']
    episode_num =0
    rews = []
    save_freq = 10

    while True:
        episode_num += 1
        total_reward = 0
        obj_poses = obj_pose_sensor.get_poses()
        target_num =  np.random.randint(0,len(targets)-1)
        target_name = targets[target_num]
        rl_place_agent.target_num = target_num
        rl_place_agent.target_name = target_name
        target_state = list(obj_poses[target_name])

        try:
            obs = manual_agent.pickup_and_stage_object(target_name, task,obj_pose_sensor)
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


            actions = rl_place_agent.act(obs,obj_poses, key=target_name)

            try:
                obs, reward, terminal = task.step(actions)
                
                ### Check current distance
                rl_place_agent.dist_after_action = max(0.05,np.linalg.norm(target_state[:3] - obs.gripper_pose[:3]))

                # Update arm states
                rl_place_agent.has_object = len(task._robot.gripper._grasped_objects) > 0
        
                obj_poses = obj_pose_sensor.get_poses()
                rl_place_agent.obj_poses = obj_poses
                rl_place_agent.gripper_pose = obs.gripper_pose

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
                break

            ## Observe results
            rl_place_agent.agent.observe(terminal=terminal, reward=reward)
            total_reward += reward

            print('Iteration: %i, Reward: %.1f' %(i, reward))
        
        print('Episode: %i, Total Reward: %.1f' %(episode_num,total_reward))
        rews.append(total_reward)
        print('Reset')
        rl_place_agent.agent.reset()
        descriptions, obs = task.reset()


        if episode_num % save_freq == save_freq - 1: rl_place_agent.agent.save(directory=save_name)
        

    # rl_place_agent.agent.close()
    env.shutdown()


