import itertools
import os
import sys

import gym
import numpy as np
import click

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib.tilecoding import representation

class Estimator():
    """
    Value Function approximator.
    """
    def __init__(self, env, weights):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        
        self.action_length = env.action_space.n

        state_range = [
            np.array(env.observation_space.low), 
            np.array(env.observation_space.high)]
        
        self.featurizer = representation.TileCoding(
            input_indices = [np.arange(env.observation_space.shape[0])], # Tiling in each dimension
            ntiles = [20], # Number of tiles per dimension
            ntilings = [5], # How many layers of tiles
            hashing = None,
            state_range = state_range)

        # After choice of feature type
        self.feature_length = self.featurizer._TileCoding__size

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.feature_length, self.action_length)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        ## polynomial features function requires 2D matrix
        # features = self.poly.fit_transform(state.reshape(1, -1))
        features = np.zeros((self.feature_length,1))
        features_index = self.featurizer(state)[0:-1]
        features[features_index] = 1
        return features, features_index
    
    def predict(self, state):
        """
        Makes value function predictions.
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
        """
        
        #1: Featurise states & action
        features, _ = self.featurize_state(state)
        #2: Calculate action-value
        value = np.matmul(features.T, self.weights).flatten()
        #3: Select return
        return value


    def update(self, eligibility, td_error, alpha):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        
        self.weights += alpha * td_error * eligibility

        return None

def make_epsilon_greedy_policy(estimator, epsilon):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A = np.ones(estimator.action_length, dtype=float) * epsilon / estimator.action_length
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def TD_lambda(env, estimator, num_episodes, epsilon_func, display=True, discount_factor=0.9,
        epsilon_decay=0.99, epsilon_initial=0.25, visual_updates=10, llambda=0.8, alpha=1e-3):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    with click.progressbar(range(num_episodes)) as bar:

        for episode in bar: 
            epsilon = epsilon_func(episode,epsilon_decay,epsilon_initial)
            observation = env.reset()
            counter = 0
            episode_tracker = []
            eligibility = np.zeros((estimator.feature_length,estimator.action_length))

            while True:

                if (episode%visual_updates == 0) & display:
                    env.render()

                counter += 1
                episode_tracker.append(observation)
                q_values = estimator.predict(observation)
                feature, feature_index = estimator.featurize_state(observation)
                actions = np.arange(0, env.action_space.n)
                policy = make_epsilon_greedy_policy(estimator, epsilon)
                probs = policy(observation)
                action = np.random.choice(actions, p=probs)
                q_value = q_values[action]

                observation_new, reward, done, _ = env.step(action)
                probs_new = policy(observation_new)
                q_values_new = estimator.predict(observation_new)

                on_policy = True

                if on_policy:
                    # e-greedy: On Policy
                    action_new = np.random.choice(np.arange(0,env.action_space.n), p=probs_new)
                    q_value_next = q_values_new[action_new]
                else:
                    # max Q value next: Off policy Q learning
                    q_value_next = np.max(q_values_new)

                td_target = reward + discount_factor * q_value_next
                td_error = td_target - q_value

                # Eligibility decay
                eligibility = discount_factor*llambda*eligibility
                eligibility[:,action] += feature.T.flatten()

                # Update weights                    
                estimator.update(eligibility, td_error, alpha)

                observation = observation_new

                # Update statistics
                stats.episode_rewards[episode] += reward  
                
                if done:
                    stats.episode_lengths[episode] = counter
                    break
    
    return stats, estimator