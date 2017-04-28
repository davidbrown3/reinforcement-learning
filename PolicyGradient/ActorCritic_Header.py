import matplotlib
import gym
import pickle

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import ValueEstimator, PolicyEstimator, ActorCritic

test = 'MountainCarContinuous-v0'

if test is "MountainCarContinuous-v0":
    method = "PolicySampledExperience"
    state_labels = ['x', 'x_dot']
    policy_alpha=2.5e-3
    policy_std=5*10**-1
    samplerange=5000
    visual_updates=25
    plot_updates=25
    policy_llambda=0
    # Higher number of tile layers seems very beneficial!
    
elif test is "Pendulum-v0":
    method = "PolicyRandomExperience"
    state_labels = ['cos(θ)', 'sin(θ)', 'θ_dot']
    policy_alpha=1e-4
    policy_std=10**-1
    samplerange=10000
    plot_updates=25
    visual_updates=25
    policy_llambda=0

env = gym.envs.make(test)

weights = None
policy_estimator = PolicyEstimator(env, weights, state_labels)
value_estimator = ValueEstimator(env, weights, state_labels)

num_episodes = 1000
stats, value_estimator, policy_estimator = ActorCritic(
    env, policy_estimator, value_estimator, num_episodes, display=True,
    policy_alpha=policy_alpha, visual_updates=visual_updates, samplemethod=method,
    policy_std=policy_std,samplerange=samplerange, plot_updates=plot_updates,
    policy_llambda=policy_llambda)

# Policy alpha linked to STD!

pickle.dump(value_estimator, open( "savedvaluemodel.txt", "wb" ) )
pickle.dump(policy_estimator, open( "savedpolicymodel.txt", "wb" ) )

# plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)