import numpy as np
import sys, os
import pprint

sys.path.append(os.path.dirname(sys.path[0]))
from lib.envs.gridworld import GridworldEnv
from ValueIteration import value_iteration

env = GridworldEnv()
pp = pprint.PrettyPrinter(indent=2)

random_policy = np.ones([env.nS, env.nA]) / env.nA

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
