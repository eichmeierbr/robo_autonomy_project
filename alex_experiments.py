import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from AutonAgent import *



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

def check_if_in_cupboard(target_name, obj_poses):
    t_pos = obj_poses[target_name[:-12]]
    t_pos = agent2.convertTargetCoordsToCabinet(t_pos)
    in_cab =            agent2.x_r[0] - 0.05 <= t_pos[0] <= agent2.x_r[1] + 0.05
    in_cab = in_cab and agent2.y_r[0] - 0.05 <= t_pos[1] <= agent2.y_r[1] + 0.05
    in_cab = in_cab and agent2.z_r[0] - 0.05 <= t_pos[2] <= agent2.z_r[1] + 0.05
    return in_cab

def resetTask(task):
    obs = task.get_observation()
    obj_poses = obj_pose_sensor.get_poses()
    #get items in cupboard
    in_cupboard = ['sugar']
    print('started reseting')
    #for k,v in obj_poses:
    #    if check_if_in_cupboard(k,obj_poses):
    #        in_cupboard.append(k)
    
    #drop anything in hand
    agent2.ungrasp_object(obs)

    #move to start position
    actions = agent2.move_to_pos([0.25, 0, 1])
    obs, reward, terminal = task.step(actions)
    print('moved to start')

    for obj in in_cupboard:
        #move to above object location
        actions = agent2.move_above_cabinet(obj_poses, obj)
        obs, reward, terminal = task.step(actions)
        print('move above cabinet')

        #attempt straight grasp
        grasped = False
        actions = agent2.move_to_cabinet_object(obj_poses, obj, False)
        prev_forces = obs.joint_forces
        print(np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01)
        print(not grasped)
        print(np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50)
        while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01 and not grasped and np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50:
            prev_forces = obs.joint_forces
            print('stepping to target')
            obs, reward, terminate = task.step(actions)

            for g_obj in task._task.get_graspable_objects():
                obj_name = g_obj.get_name()
                
                if obj_name == obj:
                    grasped = task._robot.gripper.grasp(g_obj)
                    print(obj_name, grasped)
        
        #if failed kick the object to the back of the line and try another
        if (not grasped):
            in_cupboard.append(obj)
            print('kicking')
            continue

        #remove from cabinet
        actions = agent2.move_above_cabinet_num(obs, obj_poses, 5)
        obs, reward, terminal = task.step(actions)
        print('moved above cabinet_num')

        #place on table
        print ('place on table')
        


    
    
    descriptions=None
    #original reset task
    #descriptions, obs = task.reset()
    return descriptions, obs

if __name__ == "__main__":
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes

    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    len_episode = 10

    
    # agent = TensorForceDQN()
    # agent = TensorForceClass(load='rl_models')

    # agent.len_episode = len_episode
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    
    descriptions, obs = task.reset()
    print(descriptions)
    agent2 = AutonAgentAbsolute_Mode()
    targets = ['crackers', 'mustard', 'coffee', 'sugar','spam', 'tuna', 'soup', 'strawberry_jello', 'chocolate_jello']
    episode_num =0
    rews = []
    save_freq = 20
    item='sugar'
    #localize the cupboard
    #obj_poses = obj_pose_sensor.get_poses()
    #actions = agent2.move_above_cabinet(obj_poses, 'waypoint3')
    #obs, reward, terminal = task.step(actions)
    #print ('moved above cabinet for localization')
    #actions = agent2.move_into_cabinet(obs, obj_poses, 5)
    #obs, reward, terminal = task.step(actions)
    #print ('moved into cabinet for localization')
    #spot=-.1
    #prev_forces = obs.joint_forces
    #while(np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50):
    #    prev_forces = obs.joint_forces
    #    actions = agent2.move_into_cabinet(obs, obj_poses, 5)
    #    obs, reward, terminal = task.step(actions)
    #    spot+=.005
    
    #print ('localized cabinet')
    #actions = agent2.move_above_cabinet(obj_poses, item)
    #obs, reward, terminal = task.step(actions)
    #print ('destaging localization')


    while True:
        episode_num += 1
        total_reward = 0
        obj_poses = obj_pose_sensor.get_poses()
        target_name = targets[np.random.randint(0,len(targets)-1)]

        target_state = list(obj_poses[target_name])

        ## Stage point to avoid cupboard
        actions = agent2.move_to_pos([0.25, 0, 1])
        obs, reward, terminal = task.step(actions)
        print ('moved to start')
        
        depth = 0
        while(obs.joint_forces[1]  >= -10):
            ## Stage above object
            actions = agent2.move_above_object_dep(obj_poses, item, depth)
            obs, reward, terminal = task.step(actions)
            depth += .01
        print ('moved above object')

        actions = agent2.grasp_object(obs)
        obs, reward, terminal = task.step(actions)
        print ('grasp object')

        actions = agent2.move_above_cabinet_num(obs, obj_poses, 5)
        obs, reward, terminal = task.step(actions)
        print ('moved above cabinet')

        actions = agent2.move_into_cabinet(obs, obj_poses, 5)
        obs, reward, terminal = task.step(actions)
        print ('moved into cabinet')

        actions = agent2.ungrasp_object(obs)
        obs, reward, terminal = task.step(actions)
        print ('ungrasp object')

        actions = agent2.move_above_cabinet_num(obs, obj_poses, 5)
        obs, reward, terminal = task.step(actions)
        print ('moved above cabinet num')
        
        actions = agent2.move_to_pos([0.25, 0, 1])
        obs, reward, terminal = task.step(actions)
        print ('moved to start')

        actions = agent2.move_above_cabinet_num(obs, obj_poses, 1)
        obs, reward, terminal = task.step(actions)
        print ('moved above cabinet num')

        resetTask(task)

        condition = False
        if(condition):
            sweep = 11.5-item+episode_num
            actions = agent2.move_above_cabinet(obs, obj_poses, sweep)
            obs, reward, terminal = task.step(actions)
            print ('staging sweep')

            while (sweep == 11.5-item+episode_num):
                actions = agent2.move_into_cabinet(obs, obj_poses, sweep)
                obs, reward, terminal = task.step(actions)
                print ('moved into cabinet for sweep')

                spot=0
                prev_forces = obs.joint_forces
                while(np.sum(np.abs(obs.joint_forces-prev_forces)) <= 50):
                    prev_forces = obs.joint_forces
                    actions = agent2.move_into_cabinet(obs, obj_poses, sweep-spot)
                    obs, reward, terminal = task.step(actions)
                    print ('sweeping')
                    spot+=.5

                actions = agent2.move_above_cabinet(obs, obj_poses, sweep-spot)
                obs, reward, terminal = task.step(actions)
                print ('destaging sweep')
                if spot >= 11.5-item+episode_num:
                    sweep = 0

        

#        for i in range(len_episode):
#            # Getting noisy object poses
#            obj_poses = obj_pose_sensor.get_poses()
#    
#            # Getting various fields from obs
#            current_joints = obs.joint_positions
#            gripper_pose = obs.gripper_pose
#            rgb = obs.wrist_rgb
#            depth = obs.wrist_depth
#            mask = obs.wrist_mask
#            agent.has_object = len(task._robot.gripper._grasped_objects) > 0
#    
#
#            actions = agent.act(obs,obj_poses)
#
#            try:
#                obs, reward, terminal = task.step(actions)
#    
#                ### Check current distance
##                agent.dist_after_action = np.linalg.norm(target_state[:3] - obs.gripper_pose[:3])
    
#######                ### Calculate reward
#######                reward, terminal = agent.calculateReward()
#######
#######            except:
#######                reward = -agent.dist_before_action*5
#######                terminal = False
#######
#######            ## Observe results
#######            agent.agent.observe(terminal=terminal, reward=reward)
#######            total_reward += reward
#######
#######            if terminal: 
#######                break
#######
#######            print('Iteration: %i, Reward: %.1f' %(i, reward))
#######
#######        print('Episode: %i, Average Reward: %.1f' %(episode_num,total_reward))
#######        rews.append(total_reward)
#######        print('Reset')
#        descriptions, obs = task.reset()
#######        agent.agent.reset()
#######
#######
#######        if episode_num % save_freq == save_freq - 1: agent.agent.save(directory='dqn_grasp')
        

    env.shutdown()
    agent.agent.close()


