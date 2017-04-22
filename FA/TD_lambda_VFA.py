import itertools
import os
import sys
import matplotlib
import gym
import numpy as np
import click
import time

from collections import deque

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting, EpsilonFunction
from lib.tilecoding import representation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Estimator():
    """
    Action-Value Function approximator.
    """
    def __init__(self, env, weights):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        
        self.action_length = env.action_space.n
        self.state_length = env.observation_space.shape[0]
        state_range = [
            np.array(env.observation_space.low), 
            np.array(env.observation_space.high)]
        
        self.featurizer = representation.TileCoding(
            input_indices = [np.arange(self.state_length)], # Tiling in each dimension
            ntiles = [10], # Number of tiles per dimension
            ntilings = [3], # How many layers of tiles
            hashing = None,
            state_range = state_range)
        
        # MORE TILING: share weights for neibouring states. Less to learn

        # After choice of feature type
        self.feature_length = self.featurizer._TileCoding__size

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.feature_length, self.action_length)
        
        self.weights_old = self.weights

        # To avoid singularities, initialise with small diagonal terms
        eps = 10**-3
        self.a_tracker = np.eye(self.feature_length*self.action_length)*eps
        self.b_tracker = np.zeros((self.feature_length*self.action_length, 1))

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        ## polynomial features function requires 2D matrix
        # features = self.poly.fit_transform(state.reshape(1, -1))
        t = time.time()
        features = np.zeros((self.feature_length,1))
        features_index = self.featurizer(state)[0:-1]
        features[features_index] = 1
        elapsed = time.time() - t 
        return features, features_index
    
    def predict(self, state, weights):
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
        value = np.matmul(features.T, weights).flatten()
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

    # Needs renaming to LSQ
    def update_LSDQ(self, experience, discount_factor, count, dynamic_experience=False):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """

        # Initialise policy for current time
        epsilon = 0.0
        full_length = self.feature_length * self.action_length

        with click.progressbar(range(len(experience))) as bar:
            for idx in bar:
                (observation, action, reward, observation_new) = experience[idx]
            
                probs_old = self.policy(observation_new, epsilon=epsilon, weights=self.weights_old)
                probs = self.policy(observation_new, epsilon=epsilon, weights=self.weights)

                action_policy_old = np.argmax(probs_old)
                action_policy_new = np.argmax(probs)

                if (action_policy_new == action_policy_old) & count>0:
                    continue
                
                # Faster matrix multiplication for known sparse outer product
                _, x_SA_IX = self.featurize_state(observation)
                x_SA_IX_full = x_SA_IX + action*self.feature_length

                _, x_SA_new_IX = self.featurize_state(observation_new)
                x_SA_new_add_IX_full = x_SA_new_IX + action_policy_new*self.feature_length
                x_SA_new_subtract_IX_full = x_SA_new_IX + action_policy_old*self.feature_length
                
                update_add = np.zeros((full_length, full_length))
                update_subtract = np.zeros((full_length, full_length))
                
                if (count == 0) | dynamic_experience:
                    
                    # Outer product complexity is not requred for tile coding
                    self.a_tracker[x_SA_IX_full, x_SA_IX_full] += 1 # Assumes tiles come out as 1 each
                    self.a_tracker[x_SA_IX_full, x_SA_new_add_IX_full] -= discount_factor # Assumes tiles come out as 1 each
                    
                    idx = action * self.feature_length
                    update_b = self.featurize_state(
                        observation)[0].reshape((self.feature_length,1)).flatten()

                    self.b_tracker[idx:idx+self.feature_length, 0] += update_b * reward
                    
                elif (action_policy_new != action_policy_old):

                    # Outer product complexity is not requred for tile coding
                    # Don't need to operate on diagonals as they cancel out
                    self.a_tracker[x_SA_IX_full,x_SA_new_add_IX_full] -= discount_factor # Assumes tiles come out as 1 each
                    self.a_tracker[x_SA_IX_full,x_SA_new_subtract_IX_full] += discount_factor
            
        self.weights_old = self.weights

        # Efficient to solution to inv(A)*B
        self.weights = np.linalg.solve(self.a_tracker, self.b_tracker).reshape(
            self.feature_length, self.action_length, order='F')

        return None

    def policy(self, observation, weights, epsilon=0):
        q_values = self.predict(observation, weights=weights)
        best_action = np.argmax(q_values)
        A = np.ones(self.action_length, dtype=float) * epsilon / self.action_length
        A[best_action] += (1.0 - epsilon)
        return A

# Needs renaming to LSPI
def experiment(env, estimator, discount_factor=0.99):

    reward_end = 0
    samplerange = 50000
    plotrange = 50
    dynamic_experience = False # Training data changes throughout

    samples = np.empty(shape=(samplerange, estimator.state_length))
    plots = np.empty(shape=(plotrange, estimator.state_length))
    plots = []
    for ix, (low, high) in enumerate(zip(env.observation_space.low,env.observation_space.high)):
        plots.append(np.linspace(low, high, num=50).tolist())
        samples[:,ix] = np.random.uniform(low=low, high=high, size=samplerange)
    
    experience = deque(maxlen=samplerange)
    
    # Training data from sweep of random single steps
    for ix in range(samplerange):
        env.reset()
        env.env.state = samples[ix,:]
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            reward = reward_end # Can try an alternative reward if episode is done
        experience.append((samples[ix,:],action,reward,observation))
    '''

    # Training data from random policy
    observation = env.reset()
    for ix in range(samplerange):
        action = env.action_space.sample()
        observation_new, reward, done, info = env.step(action)
        experience.append((observation, action, reward, observation_new))
        if done:
            observation = env.reset()
        else:
            observation = observation_new
    '''
    X, Y = np.meshgrid(*plots)
    XY_combos = np.array([X, Y]).T.reshape(-1, estimator.state_length)
    
    for count in range(100):
        
        estimator.update_LSDQ(experience, discount_factor, count,
            dynamic_experience=dynamic_experience)

        if (count%1)==0:
            Z = np.zeros(XY_combos.shape[0])
            A = np.zeros(XY_combos.shape[0], dtype=int)
            for Idx, state in enumerate(XY_combos):
                values = estimator.predict(state, estimator.weights)
                A[Idx] = np.argmax(values)
                Z[Idx] = values[A[Idx]]
            Z = Z.reshape(X.shape)
            A = A.reshape(X.shape)

            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm)
            fig.colorbar(surf)
            ax.view_init(ax.elev, -120)
            plt.savefig('Value'+str(count), orientation='landscape')
            plt.close(fig)

            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, A, cmap=matplotlib.cm.coolwarm)
            fig.colorbar(surf)
            ax.view_init(ax.elev, -120)
            plt.savefig('Action'+str(count), orientation='landscape')
            plt.close(fig)

        observation = env.reset()
        rewards = 0
        steps = 0
        while True:
            steps += 1
            #env.render()
            feature, feature_index = estimator.featurize_state(observation)
            actions = np.arange(0, env.action_space.n)
            probs = estimator.policy(observation, weights=estimator.weights, epsilon=0)
            action = np.random.choice(actions, p=probs)
            observation_new, reward, done, _ = env.step(action)
            rewards += reward
            observation = observation_new
            if done & (steps<env._max_episode_steps):
                reward = reward_end
                #experience.append((observation, action, reward, observation_new))
                print(rewards)
                break
            elif done:
                print(rewards)
                break
            else:
                #experience.append((observation, action, reward, observation_new))
                1

def TD_lambda(env, estimator, num_episodes, epsilon_func, display=True, discount_factor=0.9,
        epsilon_decay=0.99, epsilon_initial=0.25, visual_updates=10, llambda=0.8, alpha=1e-3,
        momentum=0.9):
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
            update = 0

            while True:

                if (episode%visual_updates == 0) & display:
                    env.render()

                counter += 1
                episode_tracker.append(observation)
                q_values = estimator.predict(observation)
                feature, feature_index = estimator.featurize_state(observation)
                actions = np.arange(0, env.action_space.n)
                probs = estimator.policy(observation, estimator.weights, epsilon)
                action = np.random.choice(actions, p=probs)
                q_value = q_values[action]

                observation_new, reward, done, _ = env.step(action)
                if done:
                    reward = 1 # Can try an alternative reward if episode is done
                
                probs_new = estimator.policy(observation_new, estimator.weights, epsilon)
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
                update = estimator.update(eligibility, td_error, alpha, update, momentum)

                observation = observation_new

                # Update statistics
                stats.episode_rewards[episode] += reward  
                
                if done:
                    stats.episode_lengths[episode] = counter
                    break
    
    return stats, estimator