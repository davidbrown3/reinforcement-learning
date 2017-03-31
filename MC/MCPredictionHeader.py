import matplotlib
import numpy as np
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from lib.envs.blackjack import BlackjackEnv
from lib import plotting
from MCPrediction import mc_prediction
from SampleBJPolicy import sample_policy

matplotlib.style.use('ggplot')

env = BlackjackEnv()

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")