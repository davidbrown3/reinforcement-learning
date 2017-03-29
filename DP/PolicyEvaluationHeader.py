import numpy as np
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from lib.envs.gridworld import GridworldEnv
from PolicyEvaluation import policy_eval

env = GridworldEnv()

random_policy = np.ones([env.nS, env.nA]) / env.nA

v = policy_eval(random_policy, env, theta=0.00001)

print(v)

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)