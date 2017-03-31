import numpy as np
import click
from collections import defaultdict

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    with click.progressbar(range(num_episodes)) as bar:
        for episode in bar:
            observation = env.reset()
            episode_tracker = []

            while True:
                action = policy(observation)
                observation_new, reward, done, _ = env.step(action)
                episode_tracker.append((observation, action, reward))
                
                if done:
                    break
                else: 
                    observation = observation_new
            
            rewards = [x[2] for x in episode_tracker]

            states_in_episode = set([tuple(x[0]) for x in episode_tracker])
            for stateIdx, state in enumerate(states_in_episode):

                total_reward = 0
                for future_stepIdx, future_step in enumerate(range(stateIdx, len(states_in_episode))):
                    total_reward += rewards[future_step] * discount_factor**future_stepIdx

                returns_count[state] += 1.0
                V[state] = V[state] + (total_reward-V[state])/returns_count[state]
    
    return V    