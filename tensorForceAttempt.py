import gym
import rlbench.gym
from tensorforce import Agent
# from tensorforce import Agent, Environment


import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *



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



def scaleActions(actions):
    x_r = [-0.0, 0.45]
    y_r = [-0.4, 0.4]
    z_r = [0.8, 1.6]

    actions[0] = actions[0]*(x_r[1] - x_r[0]) + x_r[0]
    actions[1] = actions[1]*(y_r[1] - y_r[0]) + y_r[0]
    actions[2] = actions[2]*(z_r[1] - z_r[0]) + z_r[0]

    return actions


if __name__ == "__main__":
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes

    num_states = 6
    num_inputs = 4
    input_high = 1.0
    input_low  = 0.0
    
    states_dict = {'type': 'float', 'shape': num_states}
    actions_dict = {'type': 'float', 'shape': num_inputs, 'min_value': input_low, 'max_value': input_high}

    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    len_episode = 30
    explore = 0.6



    agent = Agent.create(
        agent='dqn',
        states = states_dict,  # alternatively: states, actions, (max_episode_timesteps)
        actions = actions_dict,
        memory=10000,
        max_episode_timesteps= len_episode,
        exploration= explore
    )
 
     
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    
    descriptions, obs = task.reset()
    print(descriptions)

    

    while True:
        obj_poses = obj_pose_sensor.get_poses()
        target_state = list(obj_poses['sugar'])
        target_state[2] += 0.1

        # best_reward = 1/np.linalg.norm(target_state[:3] - obs.gripper_pose[:3])
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
            # agent.has_object = len(task._robot.gripper._grasped_objects) > 0
    

            ###########################################################
            ###### PREPARE INPUT STATES TO RL FUNCTION ################
            target_state = list(obj_poses['sugar'])
            target_state[2] += 0.1
            # in_states = list(gripper_pose)
            # in_states.extend(target_state)

            in_states = list(gripper_pose[:3])
            in_states.extend(list(target_state[:3]))
            # in_states.extend(list(obj_poses['cupboard']))
            ###### PREPARE INPUT STATES TO RL FUNCTION ################
            ###########################################################

            actions = agent.act(states= in_states)
            a_in = scaleActions(actions)

            # a_in = gripper_pose[:3] - actions[:3]
            
            actions2 = list(a_in) + [0,1,0,0] + list([actions[3]>0.5])


            try:
                obs, reward, terminal = task.step(actions2)
                reward = 0

                ## See if we're closer to the target object
                start_dist = np.linalg.norm(target_state[:3] - gripper_pose[:3])
                now_dist = np.linalg.norm(target_state[:3] - obs.gripper_pose[:3])

                if start_dist > now_dist:
                    reward = 1/now_dist
                    reward = max(reward - best_reward, 0) + 1
                    best_reward = max(reward, best_reward)

                elif start_dist < now_dist:
                    reward = 0.0

                if len(task._robot.gripper._grasped_objects) > 0: 
                    reward = 50.0
                    terminal = True

                agent.observe(terminal=terminal, reward=reward)

            except:
                reward = -0.5
                terminal = False
                agent.observe(terminal=terminal, reward=reward)

            print('Iteration: ' + str(i) + ', Reward: ' + str(reward))

        print('Reset')
        descriptions, obs = task.reset()
        

    env.shutdown()
    # environment.close()
    agent.close()


## X Range: -0.025 - 0.52
## Y Range: -0.45 - 0.45
## Z Range: 0.751 - 1.75 (Maybe a little higher)