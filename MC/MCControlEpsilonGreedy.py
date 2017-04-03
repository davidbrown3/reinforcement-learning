from collections import defaultdict

import click
import numpy as np


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon) 
        return A
        
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    with click.progressbar(range(num_episodes)) as bar:
        for episode in bar:

            observation = env.reset()
            episode_tracker = []

            while True:
                probs = policy(observation) 
                action = np.random.choice(np.arange(0,env.action_space.n), p=probs)
                observation_new, reward, done, _ = env.step(action)
                episode_tracker.append((observation, action, reward))
                
                if done:
                    break
                else: 
                    observation = observation_new
            
            rewards = [x[2] for x in episode_tracker]
            sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode_tracker])

            for Idx, sa_pair in enumerate(sa_in_episode):
                
                state = sa_pair[0]
                action = sa_pair[1]

                total_reward = 0
                for future_stepIdx, future_step in enumerate(range(Idx, len(sa_in_episode))):
                    total_reward += rewards[future_step] * discount_factor**future_stepIdx
                
                returns_count[sa_pair] += 1.0
                Q[state][action] = Q[state][action] + (total_reward-Q[state][action])/returns_count[sa_pair]
            
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    return Q, policy
