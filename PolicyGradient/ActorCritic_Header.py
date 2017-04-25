import matplotlib
import gym
import pickle

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import ValueEstimator, PolicyEstimator, ActorCritic

matplotlib.style.use('ggplot')

test = "MountainCarContinuous-v0"

env = gym.envs.make(test)

weights = None
policy_estimator = PolicyEstimator(env, weights)
value_estimator = ValueEstimator(env, weights)

num_episodes = 1000000
stats, value_estimator, policy_estimator = ActorCritic(
    env, policy_estimator, value_estimator, num_episodes, display=True,
    policy_alpha=1e-4, visual_updates=25)

# Policy alpha linked to STD!

pickle.dump(value_estimator, open( "savedvaluemodel.txt", "wb" ) )
pickle.dump(policy_estimator, open( "savedpolicymodel.txt", "wb" ) )

# plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)