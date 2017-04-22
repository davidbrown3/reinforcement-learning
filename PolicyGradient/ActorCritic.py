import itertools
import os
import sys

import gym
import numpy as np
import click

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib.tilecoding import representation

class PolicyEstimator():
    """
    Policy Function approximator.
    """
    def __init__(self, env, weights):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        
        self.action_length = 1

        state_range = [
            np.array(env.observation_space.low), 
            np.array(env.observation_space.high)]
        
        self.NTiling = 25
        self.featurizer = representation.TileCoding(
            input_indices = [np.arange(env.observation_space.shape[0])], # Tiling in each dimension
            ntiles = [20], # Number of tiles per dimension
            ntilings = [self.NTiling], # How many layers of tiles
            hashing = None,
            state_range = state_range)

        # After choice of feature type
        self.feature_length = self.featurizer._TileCoding__size

        self.weights = (env.action_space.high/self.NTiling) * np.random.normal(size=[self.feature_length, self.action_length])

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
        mean = np.matmul(features.T, self.weights).flatten()
        
        return mean


    def update(self, eligibility, alpha, action_score, update_prev, momentum):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        
        update = np.array(alpha * eligibility * action_score).reshape(-1,1) + momentum * update_prev
        self.weights += update

        return update

class ValueEstimator():
    """
    Value Function approximator.
    """
    def __init__(self, env, weights):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        
        state_range = [
            np.array(env.observation_space.low), 
            np.array(env.observation_space.high)]
        
        self.NTiling = 25
        self.featurizer = representation.TileCoding(
            input_indices = [np.arange(env.observation_space.shape[0])], # Tiling in each dimension
            ntiles = [20], # Number of tiles per dimension
            ntilings = [self.NTiling], # How many layers of tiles
            hashing = None,
            state_range = state_range)

        # After choice of feature type
        self.feature_length = self.featurizer._TileCoding__size

        self.weights = np.random.rand(self.feature_length)

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


    def update(self, eligibility, td_error, alpha, update_prev, momentum):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        
        update = alpha * td_error * eligibility + momentum * update_prev
        self.weights += update

        return update

def ActorCritic(env, policy_estimator, value_estimator, num_episodes, display=True,
        epsilon_decay=0.99, epsilon_initial=0.25, visual_updates=10, policy_std=0.1,
        value_alpha=1e-3,  value_discount_factor=0.9,  value_llambda=0.8, value_momentum=0.9,
        policy_alpha=1e-3, policy_discount_factor=0.9, policy_llambda=0, policy_momentum=0):
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
            observation = env.reset()
            counter = 0
            episode_tracker = []
            value_eligibility = np.zeros(value_estimator.feature_length)
            policy_eligibility = np.zeros(policy_estimator.feature_length)
            value_update = 0
            policy_update = 0

            while True:

                if (episode%visual_updates == 0) & display:
                    env.render()

                counter += 1
                episode_tracker.append(observation)

                policy_feature, _ = policy_estimator.featurize_state(observation)
                mean_action = policy_estimator.predict(observation)
                action = np.random.normal(loc=mean_action, scale=policy_std)
                policy_score = ((action-mean_action) * policy_feature / (policy_std**2)).flatten() # See DSilver notes

                observation_new, reward, done, _ = env.step(action)

                # Value TD error
                value = value_estimator.predict(observation)
                value_feature, _ = value_estimator.featurize_state(observation)
                value_new = value_estimator.predict(observation_new)
                value_td_target = reward + value_discount_factor * value_new
                value_td_error = value_td_target - value

                # Eligibility decay
                value_eligibility = value_discount_factor * value_llambda * value_eligibility
                value_eligibility += value_feature.T.flatten()/value_estimator.NTiling
                policy_eligibility = policy_discount_factor * policy_llambda * policy_eligibility
                policy_eligibility += policy_score.flatten()/policy_estimator.NTiling

                # Update weights                    
                value_update = value_estimator.update(
                    value_eligibility, value_td_error, value_alpha, value_update, value_momentum)
                policy_update = policy_estimator.update(
                    policy_eligibility, policy_alpha, policy_score, policy_update, policy_momentum)

                # Update statistics
                stats.episode_rewards[episode] += reward  
                
                # Loop back parameters
                observation = observation_new

                if done:
                    stats.episode_lengths[episode] = counter
                    print(stats.episode_rewards[episode])
                    break
    
    return stats, value_estimator, policy_estimator