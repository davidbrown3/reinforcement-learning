import numpy as np
import sys, os
import itertools
import matplotlib
import pandas as pd
import click
from collections import defaultdict

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting

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

def sarsa(env, num_episodes, epsilon_func, discount_factor=1.0, alpha=0.5, epsilon=0.1, llambda=0.8):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Eligibility = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    with click.progressbar(range(num_episodes)) as bar:
        for episode in bar:

            epsilon = epsilon_func(episode)
            observation = env.reset()
            counter = 0
            episode_tracker = []

            # Reset eligibility
            for observation_key in Eligibility.keys(): 
                for actionIdx, action_key in enumerate(Eligibility[observation]):
                    Eligibility[observation_key][actionIdx] = 0

            while True:
                episode_tracker.append(observation)
                counter += 1
                policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
                probs = policy(observation)
                action = np.random.choice(np.arange(0,env.action_space.n), p=probs)
                Eligibility[observation][action] += 1
                observation_new, reward, done, _ = env.step(action)
                
                probs_new = policy(observation_new)
                action_new = np.random.choice(np.arange(0,env.action_space.n), p=probs_new) # DB: Why use episilon greedy rather than Q greedy on 2nd step?       
                td_error = reward + discount_factor*Q[observation_new][action_new] - Q[observation][action]

                for observation_key in Eligibility.keys(): 
                    for actionIdx, _ in enumerate(Eligibility[observation_key]):
                        Eligibility[observation_key][actionIdx] = discount_factor*llambda*Eligibility[observation_key][actionIdx]
                        Q[observation_key][actionIdx] += alpha * td_error * Eligibility[observation_key][actionIdx]
                
                observation = observation_new

                # Update statistics
                stats.episode_rewards[episode] += reward  

                if done:
                    stats.episode_lengths[episode] = counter
                    break
    
    return Q, stats