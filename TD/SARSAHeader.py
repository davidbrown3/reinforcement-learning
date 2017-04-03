import gym
import matplotlib
import sys, os
from SARSA import sarsa 
from EpsilonFunction import SampleEpsilonFunction

sys.path.append(os.path.dirname(sys.path[0]))
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = WindyGridworldEnv()
Q, stats = sarsa(env, 200, SampleEpsilonFunction, discount_factor=0.99, alpha=0.5, epsilon=0.1, llambda=0.8)
plotting.plot_episode_stats(stats)