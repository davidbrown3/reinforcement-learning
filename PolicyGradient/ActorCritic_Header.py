import matplotlib
import gym
import pickle

from ActorCritic import PolicyEstimator, ValueEstimator, ActorCritic

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction

matplotlib.style.use('ggplot')

test = "MountainCarContinuous-v0"

env = gym.envs.make(test)

weights = None
policy_estimator = PolicyEstimator(env, weights)
value_estimator = ValueEstimator(env, weights)
stats, value_estimator, policy_estimator = ActorCritic(
    env, policy_estimator, value_estimator, 2000, display=True,
    value_alpha = 1e-3, policy_alpha=1e-9)

pickle.dump(value_estimator, open( "savedvaluemodel.txt", "wb" ) )
pickle.dump(policy_estimator, open( "savedpolicymodel.txt", "wb" ) )

# plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)