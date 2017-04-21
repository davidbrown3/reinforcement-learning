import matplotlib
import gym
import pickle

from TD_lambda_VFA import Estimator, TD_lambda, experiment

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction

matplotlib.style.use('ggplot')

test = "MountainCar-v0"
#test = "CartPole-v1"

env = gym.envs.make(test)

loadmodel = False

if loadmodel:
    weights = pickle.load( open( "savedmodel.txt", "rb" ) )
    estimator = Estimator(env, weights)
    # stats, estimator = TD_lambda(env, estimator, 10000, EpsilonFunction.ZeroEpsilonFunction, display=True, alpha=1e-5)

else:
    weights = None
    estimator = Estimator(env, weights)
    experiment(env, estimator)
    # stats, estimator = TD_lambda(env, estimator, 10000, EpsilonFunction.DecayEpsilonFunction, display=False, alpha=1e-3)

weights = estimator.weights
pickle.dump(weights, open( "savedmodel.txt", "wb" ) )

# plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)