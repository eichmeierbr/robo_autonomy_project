import gym
import rlbench.gym

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env = gym.make('put_groceries_in_cupboard-state-v0')
# env = gym.make('reach_target-state-v0')

training_steps = 1000
# training_steps = 600
episode_length = 40


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)


for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    action, _states = model.predict(obs)
    obs, reward, terminate, _ = env.step(action)
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()



