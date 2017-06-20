import matplotlib
import gym
import pickle

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import TileCodeEstimator, D_AC_OffP_PG, AdvantageEstimator

visual_updates = 25
plot_updates = 50
policy_std = 5*10**-1

test = 'Pendulum-v0'

if test is "MountainCarContinuous-v0":
    method = "PolicySampledExperience"
    state_labels = ['x', 'x_dot']
    observe_scalars = 1
    true_scalars = 1
    policy_llambda = 0
    policy_NLayers = 5
    policy_NTiles = 10
    value_NLayers = 5
    value_NTiles = 10
    alpha_w = 1e-2 
    alpha_v = 1e-2
    alpha_theta = 1e-3
    # Higher number of tile layers seems very beneficial!

elif test is "Pendulum-v0":
    method = "PolicyRandomExperience"
    state_labels = ['cos(θ)', 'sin(θ)', 'θ_dot']
    observe_scalars = [1, 1, 8]
    true_scalars = [1, 8]
    policy_NLayers = 5
    policy_NTiles = 10
    value_NLayers = 5
    value_NTiles = 10
    alpha_w = 1e-3 
    alpha_v = 1e-3
    alpha_theta = 1e-4

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
policy_estimator, value_estimator, advantage_estimator = D_AC_OffP_PG(env, policy_estimator, value_estimator, 
    advantage_estimator, num_episodes, method, batch_size=10000, samplerange=1000000, alpha_w=alpha_w, alpha_v=alpha_v,
     alpha_theta=alpha_theta, visual_updates=visual_updates, observe_scalars=observe_scalars, true_scalars=true_scalars,
     plot_updates=plot_updates)

# Policy alpha linked to STD!
pickle.dump(advantage_estimator, open( test + "savedadvantagemodel.txt", "wb" ) )
pickle.dump(value_estimator, open( test + "savedvaluemodel.txt", "wb" ) )
pickle.dump(policy_estimator, open( test + "savedpolicymodel.txt", "wb" ) )