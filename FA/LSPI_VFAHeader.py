import matplotlib
import gym
import pickle

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import TileCodeEstimator, TD_lambda, LSPI

matplotlib.style.use('ggplot')

test = "MountainCar-v0"
state_labels = ['x', 'x_dot']
NLayers = 5
NTiles = 10

env = gym.envs.make(test)

weights = None
estimator = TileCodeEstimator(env, weights, state_labels, NLayers=NLayers, NTiles=NTiles)
estimator = LSPI(env, estimator)

weights = estimator.weights
pickle.dump(weights, open( "savedmodel.txt", "wb" ) )
