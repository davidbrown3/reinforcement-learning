import matplotlib
import gym
import pickle

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import TileCodeEstimator, Stoc_AC_OnP_PG

test = 'MountainCarContinuous-v0'

if test is "MountainCarContinuous-v0":
    method = "PolicySampledExperience"
    state_labels = ['x', 'x_dot']
    policy_alpha = 1e-3
    policy_std = 1*10**-1
    samplerange = 5000
    visual_updates = 25
    plot_updates = 25
    policy_llambda = 0
    sample_scalars = 1
    policy_NLayers = 5
    policy_NTiles = 10
    value_NLayers = 5
    value_NTiles = 10
    # Higher number of tile layers seems very beneficial!

elif test is "Pendulum-v0":
    method = "PolicyRandomExperience"
    state_labels = ['cos(θ)', 'sin(θ)', 'θ_dot']
    policy_alpha = 1e-2
    policy_std = 10*10**-1
    samplerange = 20000
    plot_updates = 25
    visual_updates = 25
    policy_llambda = 0
    sample_scalars = [1, 1, 8]
    policy_NLayers = 5
    policy_NTiles = 10
    value_NLayers = 5
    value_NTiles = 10

'''
1) Increase corseness of value model with time?
2) env.reset random selection doesn't explore full range, so cannot be used for sweeps
3) Generate random samples and pick distribution from that?
4) Difference in evs is difference in how to get the sweep of samples?
5) Tile coding doesn't generalise or extrapolate well
'''

env = gym.envs.make(test)

weights = None
policy_estimator = TileCodeEstimator(env, weights, state_labels, NLayers=policy_NLayers, NTiles=policy_NTiles)
value_estimator = TileCodeEstimator(env, weights, state_labels, NLayers=value_NLayers, NTiles=value_NTiles)

num_episodes = 1000
value_estimator, policy_estimator = Stoc_AC_OnP_PG(
    env, policy_estimator, value_estimator, num_episodes, display=True,
    policy_alpha=policy_alpha, visual_updates=visual_updates, samplemethod=method,
    policy_std=policy_std,samplerange=samplerange, plot_updates=plot_updates,
    policy_llambda=policy_llambda, sample_scalars=sample_scalars)

# Policy alpha linked to STD!

pickle.dump(value_estimator, open( "savedvaluemodel.txt", "wb" ) )
pickle.dump(policy_estimator, open( "savedpolicymodel.txt", "wb" ) )