import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_quat_array
from scipy.spatial.transform import Rotation as R

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


class HonestObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True,
                                                                            first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            obj_poses[name] = pose
        return obj_poses


if __name__ == "__main__":
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)  # See rlbench/action_modes.py for other action modes

    env = Environment(action_mode, '', ObservationConfig(), False)
    # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    task = env.get_task(PutGroceriesInCupboard)

    agent = AutonAgentAbsolute_Mode()

    obj_pose_sensor = HonestObjectPoseSensor(env)

    descriptions, obs = task.reset()
    print(descriptions)

    # Go to staging location
    actions = [0.25, 0, 0.99, 0, 1, 0, 0, 0]
    while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01:
        print('stepping to staging')
        obs, reward, terminate = task.step(actions)
    print('moved to staging position', actions, obs.gripper_pose)

    # TODO: Get desired object from descriptions
    target_name = 'sugar_grasp_point'
    target_obj_name = 'sugar'

    obj_poses = obj_pose_sensor.get_poses()

    actions = agent.move_above_object(obj_poses, target_name, False)
    while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01:
        print('stepping to above')
        obs, reward, terminate = task.step(actions)
    print('moved above target', actions, obs.gripper_pose)

    obj_poses = obj_pose_sensor.get_poses()

    target_state = list(obj_poses[target_name])
    print(target_state)

    grasped = False

    actions = agent.move_to_pos(target_state, False)
    while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01 and not grasped:
        print('stepping to target')
        obs, reward, terminate = task.step(actions)

        for g_obj in task._task.get_graspable_objects():
            obj_name = g_obj.get_name()
            if obj_name == target_obj_name:
                grasped = task._robot.gripper.grasp(g_obj)
                print(obj_name, grasped)

    print('moved to target pos', actions, obs.gripper_pose)

    # Go to post-grasp location
    actions = [0.25, 0, 0.99, 0, 1, 0, 0, 0]
    while np.linalg.norm(obs.gripper_pose - actions[:-1]) > 0.01:
        print('stepping to post-grasp staging')
        obs, reward, terminate = task.step(actions)
    print('moved to post-grasp location', actions, obs.gripper_pose)

    env.shutdown()
