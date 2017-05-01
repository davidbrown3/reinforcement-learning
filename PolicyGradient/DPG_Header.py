import matplotlib
import gym
import pickle

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import TileCodeEstimator, D_AC_OffP_PG, AdvantageEstimator

test = 'MountainCarContinuous-v0'

if test is "MountainCarContinuous-v0":
    method = "PolicySampledExperience"
    state_labels = ['x', 'x_dot']
    policy_alpha = 2.5e-3
    policy_std = 5*10**-1
    samplerange = 5000
    visual_updates = 25
    plot_updates = 25
    policy_llambda = 0
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
advantage_estimator = AdvantageEstimator(env, weights, policy_estimator.feature_length)

num_episodes = 1000
D_AC_OffP_PG(env, policy_estimator, value_estimator, advantage_estimator, num_episodes,
    method, batch_size=10000, samplerange=1000000, alpha_w=1e-2, alpha_v=1e-2, alpha_theta=1e-3,visual_updates=250)

# Policy alpha linked to STD!

pickle.dump(value_estimator, open( "savedvaluemodel.txt", "wb" ) )
pickle.dump(policy_estimator, open( "savedpolicymodel.txt", "wb" ) )