
import gym
import matplotlib
import numpy as np
import sys, os
from MCControlEpsilonGreedy import make_epsilon_greedy_policy, mc_control_epsilon_greedy
from collections import defaultdict

sys.path.append(os.path.dirname(sys.path[0]))
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")