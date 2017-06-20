import itertools
import os
import sys
import gym
import numpy as np
import click
import time
import plotly.offline as py
import plotly.graph_objs as go
import pickle
import math
import tflearn

import tensorflow as tf

from collections import deque
from operator import itemgetter
from timeit import default_timer as timer

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting, EpsilonFunction
from lib.tilecoding import representation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NN_DDPG_Estimator():
    def __init__(self, env, sess, state_labels, actor_NLayers=[100,100], critic_NLayers=[100,100], 
        actor_alpha=0.0001, critic_alpha=0.001, tau=0.001, mini_batch=32):

        # Inputs to persistent
        self.sess = sess
        self.tau = tau
        self.actor_alpha = actor_alpha
        self.actor_NLayers = actor_NLayers
        self.critic_alpha = critic_alpha
        self.critic_NLayers = critic_NLayers
        self.state_labels = state_labels
        self.env = env
        self.mini_batch = mini_batch

        # Parsing environment
        self.action_length = 1
        self.state_length = env.observation_space.shape[0]
        self.action_range = env.action_space.high
        self.state_range = env.observation_space.high - env.observation_space.low

        # Creating models
        self.critic_states, self.critic_actions, self.critic_values = self.critic_create_network()
        self.critic_network_params = tf.trainable_variables()
        self.N_critic = len(self.critic_network_params)
        self.actor_states, _, self.actor_actions = self.actor_create_network()
        self.actor_network_params = tf.trainable_variables()[self.N_critic:]
        self.N_actor = len(self.actor_network_params)
        self.critic_target_states, self.critic_target_actions, self.critic_target_values = self.critic_create_network()
        self.critic_target_network_params = tf.trainable_variables()[(self.N_critic+self.N_actor):]
        self.actor_target_states, _, self.actor_target_actions = self.actor_create_network()
        self.actor_target_network_params = tf.trainable_variables()[(2*self.N_critic+self.N_actor):]

        # DDPG Functions
        self.predicted_q_values = tf.placeholder(tf.float32, [None, 1])
        self.critic_loss = tflearn.mean_square(self.predicted_q_values, self.critic_values) # Reduces to a single mean
        self.critic_optimize = tf.train.AdamOptimizer(self.critic_alpha).minimize(self.critic_loss)
        self.critic_action_gradients = tf.gradients(self.critic_values, self.critic_actions)
        self.critic_action_gradients_ph = tf.placeholder(tf.float32, [None, self.action_length])
        self.actor_gradients = tf.gradients(self.actor_actions, self.actor_network_params, -self.critic_action_gradients_ph)
        self.actor_optimize = tf.train.AdamOptimizer(self.actor_alpha).apply_gradients(
            zip(self.actor_gradients, self.actor_network_params))
        self.update_critic_target_network_params = \
            [self.critic_target_network_params[i].assign(
                    tf.multiply(self.critic_network_params[i], self.tau) + 
                    tf.multiply(self.critic_target_network_params[i], 1. - self.tau))
                for i in range(self.N_critic)]
        self.update_actor_target_network_params = \
            [self.actor_target_network_params[i].assign(
                    tf.multiply(self.actor_network_params[i], self.tau) + 
                    tf.multiply(self.actor_target_network_params[i], 1. - self.tau))
                for i in range(self.N_actor)]
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        
        self.save_model()

    def save_model(self):
        # Floydhub exclusively uses the 'output' directory to pass saved data back to user
        os.makedirs('output', exist_ok=True)
        self.saver.save(self.sess, 'output/tensorflow_weights.ckpt')
    def critic_gradients(self, states, actions):
        return self.sess.run(self.critic_action_gradients, feed_dict={
            self.critic_states: states,
            self.critic_actions: actions
        })
    def plot_policy_value(self, states):
        actions = self.predict_actor(states)
        return self.predict_critic(states, actions)
    def predict_actor(self, states):
        return self.sess.run(self.actor_actions, feed_dict={
            self.actor_states: states
        })
    def predict_target_actor(self, states):
        return self.sess.run(self.actor_target_actions, feed_dict={
            self.actor_target_states: states
        })
    def predict_critic(self, states, actions):
        return self.sess.run(self.critic_values, feed_dict={
            self.critic_states: states,
            self.critic_actions: actions,
        }) 
    def predict_target_critic(self, states, actions):
        return self.sess.run(self.critic_target_values, feed_dict={
            self.critic_target_states: states,
            self.critic_target_actions: actions,
        })  
    def critic_create_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_length])
        action = tflearn.input_data(shape=[None, self.action_length])
        layer1 = tflearn.fully_connected(inputs, self.critic_NLayers[0], activation='relu')
        action_layer1 = tflearn.layers.merge_ops.merge([layer1, action],'concat')
        layer2 = tflearn.fully_connected(action_layer1, self.critic_NLayers[1], activation='relu')
        outputs = tflearn.fully_connected(layer2, 1, 
            weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
        return inputs, action, outputs
    def actor_create_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_length])
        layer1 = tflearn.fully_connected(inputs, self.actor_NLayers[0], activation='relu')
        layer2 = tflearn.fully_connected(layer1, self.actor_NLayers[1], activation='relu')
        outputs = tflearn.fully_connected(layer2, self.action_length, activation='tanh',
            weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
        scaled_outputs = tf.multiply(outputs, self.action_range)
        return inputs, outputs, scaled_outputs
    def train_target(self):
        self.sess.run([self.update_critic_target_network_params, self.update_actor_target_network_params])
    def train_critic(self, states, actions, predicted_q_values):
        return self.sess.run([self.critic_optimize], feed_dict={
            self.critic_states: states,
            self.critic_actions: actions,
            self.predicted_q_values: predicted_q_values
        })
    def train_actor(self, states, gradients):
        self.sess.run(self.actor_optimize, feed_dict={
            self.actor_states: states,
            self.critic_action_gradients_ph: gradients
        })
    def old():
        '''
        # Set session
        kk.backend.set_session(sess)

        # Broken keras tings
        self.states = kk.layers.Input(shape=(self.state_length,), name="input_states")
        self.states_actor = kk.layers.Input(shape=(self.state_length,), name="input_states_actor")
        self.actions = kk.layers.Input(shape=(self.action_length,), name="input_actions")
        self.critic  = self.critic_create_network()
        self.critic_target = self.critic_create_network()
        self.actor = self.actor_create_network()
        self.actor_target = self.actor_create_network()
        #self.critic_grads = tf.multiply(kk.backend.gradients(self.critic.outputs, self.actions), 1/self.mini_batch)
        self.critic_grads = kk.backend.gradients(self.critic.outputs, self.actions)
        self.critic_grads_ph = tf.placeholder(tf.float32, [None, self.action_length])
        self.actor_grads = tf.gradients(self.actor.outputs, self.actor.trainable_weights, -self.critic_grads_ph)
        self.actor_optimize = tf.train.AdamOptimizer(actor_alpha).apply_gradients(
            zip(self.actor_grads, self.actor.trainable_weights))
        
        ## Tests pt1
        test = np.zeros(self.state_length)
        out1a = self.predict_actor(test, self.actor)
        out1b = self.predict_actor(test, self.actor_target)
        # Training target model
        self.train_target(self.critic, self.critic_target, tau=1)
        self.train_target(self.actor, self.actor_target, tau=1)
        ## Tests pt2
        out2a = self.predict_actor(test, self.actor)
        out2b = self.predict_actor(test, self.actor_target)
        # Do some assertion that out2a==out2b & out1a!=out1b

        def plot_policy_value(self, state):
            action = self.actor.predict(state.reshape(1,-1))*self.action_range
            return self.critic.predict([state.reshape(1,-1), action.reshape(1,-1)])
        def plot_actor(self, state):
            return self.actor.predict(state.reshape(1,-1))*self.action_range
        def predict_actor(self, state, model=None):
            if not model: model = self.actor
            actions = model.predict(state.reshape((-1, self.state_length)))*self.action_range
            return actions
        def actor_create_network_old(self):
            layer_1 = kk.layers.Dense(
                units=self.actor_NLayers[0],
                activation='relu',
            )(self.states_actor)
            layer_2 = kk.layers.Dense(
                units=self.actor_NLayers[1],
                activation='relu',
            )(layer_1)
            output_unscaled = kk.layers.Dense(
                units=self.action_length,
                activation='tanh',
                kernel_initializer=kk.initializers.RandomUniform(minval=-0.003, maxval=0.003)
            )(layer_2) # tanh assumes symetric distribution
            model = kk.models.Model(inputs=self.states_actor, outputs=output_unscaled)
            model.compile(loss='mse', optimizer=kk.optimizers.sgd(lr=self.actor_alpha))
            return model
        def critic_create_network_old(self):
            layer_1 = kk.layers.Dense(
                units=self.critic_NLayers[0],
                activation='relu',
            )(self.states)
            layer_2 = kk.layers.Dense( #layer_2a
                units=self.critic_NLayers[1],
                activation='relu',
            )(kk.layers.concatenate([layer_1, self.actions]))
            output = kk.layers.Dense(
                units=1,
                activation='linear',
                kernel_initializer=kk.initializers.RandomUniform(minval=-0.003, maxval=0.003)
            )(layer_2)
            model = kk.models.Model(inputs=[self.states, self.actions], outputs=output)
            model.compile(loss='mse', optimizer=kk.optimizers.adam(lr=self.critic_alpha)) #sgd
            return model
        def train_target_old(self, live, target, tau=1):
            weights = live.get_weights()
            target_weights = target.get_weights()
            for i in range(len(weights)):
                target_weights[i] = tau * weights[i] + (1 - tau)* target_weights[i]
            target.set_weights(target_weights)
        def train_actor_old(self, states, critic_gradients):
            self.sess.run(self.actor_optimize, feed_dict={
                self.states_actor: states,
                self.critic_grads_ph: critic_gradients
            })
        def train_critic_old(self, state, action, targets): 
            #self.critic.train_on_batch([state, action], targets)
            self.critic.fit([state, action], targets, verbose=0)
        def critic_gradients(self, states, actions):
            return self.sess.run(self.critic_grads, feed_dict={
                self.states: states,
                self.actions: actions
            })
        '''
class AdvantageEstimator():
    def __init__(self, env, weights, policy_feature_length):
        try:
            self.action_length = env.action_space.n
        except:
            self.action_length = 1

        self.weights = np.random.randn(policy_feature_length, self.action_length)

    def featurize_state(self, state, policy_feature, action, mean_action):
        # For a linear policy, dPolicy/dWeights = the features
        feature = policy_feature*(action-mean_action)
        return feature
    def update_SGD(self, feature, td_error, alpha, update_prev=0, momentum=0):
        update =  np.array(alpha * td_error * feature).reshape(-1,1) + momentum * update_prev
        self.weights += update
        return update
    def predict(self, feature, weights):
        value = np.matmul(feature.T, weights).flatten()
        return value

class TileCodeEstimator():
    """
    Action-Value Function approximator.
    """
    def __init__(self, env, weights, state_labels, NLayers=5, NTiles=10):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        
        self.state_labels = state_labels

        try:
            self.action_length = env.action_space.n
        except:
            self.action_length = 1

        self.state_length = env.observation_space.shape[0]
        state_range = [
            np.array(env.observation_space.low), 
            np.array(env.observation_space.high)]
        
        self.NTiling = NLayers
        self.featurizer = representation.TileCoding(
            input_indices = [np.arange(self.state_length)], # Tiling in each dimension
            ntiles = [NTiles], # Number of tiles per dimension
            ntilings = [self.NTiling], # How many layers of tiles
            hashing = None,
            state_range = state_range)
        
        # MORE TILING: share weights for neibouring states. Less to learn

        # After choice of feature type
        self.feature_length = self.featurizer._TileCoding__size

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(self.feature_length, self.action_length)
        
        self.weights_old = self.weights

        # To avoid singularities, initialise with small diagonal terms
        eps = 10**-3
        self.a_tracker = np.eye(self.feature_length*self.action_length)*eps
        self.a_tracker_inv = np.eye(self.feature_length*self.action_length)*eps
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
    
    def predict(self, state, weights=None):
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

        # Default weights
        if not weights: weights = self.weights

        #1: Featurise states & action
        features, _ = self.featurize_state(state)
        #2: Calculate action-value
        value = np.matmul(features.T, weights).flatten()
        #3: Select return
        return value
    def predict_Gaussian(self, state, weights, std):
        #1: Featurise states & action
        features, _ = self.featurize_state(state)
        #2: Calculate action-value
        mean = np.matmul(features.T, weights).flatten()
        chosen = np.random.normal(loc=mean, scale=std)
        policy_score = ((chosen-mean) * features / (std**2)).flatten() # See DSilver notes
        #3: Select return
        return mean, chosen, policy_score

    def update_SGD(self, eligibility, td_error, alpha, update_prev=0, momentum=0):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        update =  np.array(alpha * td_error * eligibility).reshape(-1,1) + momentum * update_prev
        self.weights += update
        return update
    
    def update_LSTD(self, env, experience, discount_factor, dynamic_experience=False):
        
        eps = 10**-3
        self.a_tracker = np.eye(self.feature_length*self.action_length)*eps
        self.b_tracker = np.zeros((self.feature_length*self.action_length, 1))

        reward_end = 0
        rewards = []
        with click.progressbar(range(len(experience))) as bar:
            for idx in bar:
                (observation, _, reward, observation_new, done) = experience[idx]
                x_SA, x_SA_IX = self.featurize_state(observation)
                x_SA_new, x_SA_new_IX = self.featurize_state(observation_new)
                rewards.append(reward)
                # Outer product complexity is not requred for tile coding
                mix1 = np.array(np.meshgrid(x_SA_IX, x_SA_IX)).reshape(2,-1)
                mix2 = np.array(np.meshgrid(x_SA_IX, x_SA_new_IX)).reshape(2,-1)
                self.a_tracker[np.vsplit(mix1,2)] += 1 # Assumes tiles come out as 1 each
                if not done:
                    self.a_tracker[np.vsplit(mix2,2)] -= discount_factor # Assumes tiles come out as 1 each
                self.b_tracker += reward * x_SA
        
        # Efficient to solution to inv(A)*B
        self.weights = np.linalg.solve(self.a_tracker, self.b_tracker).reshape(
            self.feature_length, self.action_length, order='F')
        
        return None

    def update_LSQ(self, experience, discount_factor, count, dynamic_experience=False):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """

        # Initialise policy for current time
        epsilon = 0.0
        full_length = self.feature_length * self.action_length
        # sherman_morrison = True

        '''
        NO NEED TO RECALCULATE FEATURES EVERY TIME! 
        WASTE OF CPU
        '''

        with click.progressbar(range(len(experience))) as bar:
            for idx in bar:
                (observation, action, reward, observation_new, done) = experience[idx]
                
                _, x_SA_IX = self.featurize_state(observation)
                _, x_SA_new_IX = self.featurize_state(observation_new)

                ''' SHOULD USE DEEP  COPY INSTEAD OF PASSING AROUND WEIGHTS
                '''

                probs_old = self.policy(observation_new, epsilon=epsilon, weights=self.weights_old)
                probs = self.policy(observation_new, epsilon=epsilon, weights=self.weights)

                action_policy_old = np.argmax(probs_old)
                action_policy_new = np.argmax(probs)

                if (action_policy_new == action_policy_old) & count>0:
                    continue
                
                # Faster matrix multiplication for known sparse outer product
                x_SA_IX_full = x_SA_IX + action*self.feature_length
                x_SA_new_add_IX_full = x_SA_new_IX + action_policy_new*self.feature_length
                x_SA_new_subtract_IX_full = x_SA_new_IX + action_policy_old*self.feature_length
                
                if (count == 0) | dynamic_experience:
                    
                    # Outer product complexity is not requred for tile coding
                    mix1 = np.array(np.meshgrid(x_SA_IX_full, x_SA_IX_full)).reshape(2,-1)
                    mix2 = np.array(np.meshgrid(x_SA_IX_full, x_SA_new_add_IX_full)).reshape(2,-1)
 
                    self.a_tracker[np.vsplit(mix1, 2)] += 1 # Assumes tiles come out as 1 each
                    
                    if not done:
                        self.a_tracker[np.vsplit(mix2, 2)] -= discount_factor # Assumes tiles come out as 1 each

                    idx = action * self.feature_length
                    update_b = self.featurize_state(
                        observation)[0].reshape((self.feature_length,1)).flatten()

                    self.b_tracker[idx:idx+self.feature_length, 0] += update_b * reward
                    
                elif (action_policy_new != action_policy_old):

                    # Outer product complexity is not requred for tile coding
                    # Don't need to operate on diagonals as they cancel out
                    mix1 = np.array(np.meshgrid(x_SA_IX_full, x_SA_new_add_IX_full)).reshape(2,-1)
                    mix2 = np.array(np.meshgrid(x_SA_IX_full, x_SA_new_subtract_IX_full)).reshape(2,-1)

                    self.a_tracker[np.vsplit(mix1,2)] -= discount_factor # Assumes tiles come out as 1 each
                    self.a_tracker[np.vsplit(mix2,2)] += discount_factor
            
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
def PolicyExperienceBatch(env, predict_function, samples, true_samples, policy_std=0):
    
    experience = deque()
    mean_actions = predict_function(samples)
    # Training data from sweep of random single steps
    for ix in range(len(samples)):
        env.reset()
        env.env.state = true_samples[ix,:]
        # action = np.random.normal(loc=mean_action, scale=policy_std)
        action = np.random.uniform(low=mean_actions[ix]-policy_std, high=mean_actions[ix]+policy_std)
        observation, reward, done, info = env.step(action)
        experience.append((samples[ix,:], action, reward, observation.flatten(), done))
    
    return experience
def PolicyExperience(env, predict_function, samples, true_samples, policy_std=0):
    
    experience = deque()
    
    # Training data from sweep of random single steps
    for ix in range(len(samples)):
        env.reset()
        env.env.state = true_samples[ix,:]
        mean_action = predict_function(samples[ix,:].reshape(1,-1)) # DIFFERENT FEATURES
        action = np.random.normal(loc=mean_action, scale=policy_std)
        observation, reward, done, info = env.step(action)
        experience.append((samples[ix,:], action, reward, observation.flatten(), done))
    
    return experience

def RandomExperience(env, samples, true_samples):

    experience = deque()

    # Training data from sweep of random single steps
    actions = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=len(samples))
    for ix in range(len(samples)):
        env.reset()
        env.env.state = true_samples[ix,:]
        action = [actions[ix]]
        observation, reward, done, info = env.step(action)
        experience.append((samples[ix,:], action, reward, observation, done))

    return experience

def PlotAction(env, predict, count=0, plotrange = 50):
    plots = []
    for ix, (low, high) in enumerate(zip(env.observation_space.low,env.observation_space.high)):
        plots.append(np.linspace(low, high, num=50).tolist())

    X, Y = np.meshgrid(*plots)
    XY_combos = np.array([X, Y]).T.reshape(-1, env.observation_space.shape[0])

    Z = np.zeros(XY_combos.shape[0])
    for Idx, state in enumerate(XY_combos):
        Z[Idx] = predict(state.reshape(1,-1))
    Z = Z.reshape(X.shape).T

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm)
    fig.colorbar(surf)
    ax.view_init(ax.elev, -120)
    plt.savefig('AC_Action'+str(count), orientation='landscape')
    plt.close(fig)    

def PlotPC(env, predictors, state_labels, count=0, observe_scalars=1, labels = ['']):
    
    data = []

    num = 5000
    combos = np.zeros((num, env.observation_space.shape[0]))
    for ix in range(num):
         combos[ix,:] = env.reset()*observe_scalars
    
    V = np.zeros((combos.shape[0], len(predictors)))
    for est_ix, predict in enumerate(predictors):
        for combo_ix, combo in enumerate(combos):
            V[combo_ix, est_ix] = predict(combo.reshape(1,-1))
        data.append(dict(range = [np.min(V[:, est_ix]), np.max(V[:, est_ix])], label = labels[est_ix], values = V[:, est_ix]))

    mins = env.observation_space.low
    maxs = env.observation_space.high
    
    for state_ix in range(env.observation_space.shape[0]):
        data.append(
            dict(
                range = [mins[state_ix], maxs[state_ix]], 
                label = state_labels[state_ix],
                values = combos[:, state_ix]
                )
            )         

    PCdata = [
        go.Parcoords(
            line = dict(color = V[:,0],
                   colorscale = 'Portland',
                   showscale = True,
                   reversescale = True,
                   cmin = np.min(V[:,0]),
                   cmax = np.max(V[:,0])),
            dimensions = data
        )
    ]

    py.plot(PCdata, filename = 'PC' + labels[0] + '_' + str(count) + '.html')

def PlotValue(env, predict, count=0, plotrange = 50):
    plots = []
    for ix, (low, high) in enumerate(zip(env.observation_space.low,env.observation_space.high)):
        plots.append(np.linspace(low, high, num=50).tolist())

    X, Y = np.meshgrid(*plots)
    XY_combos = np.array([X, Y]).T.reshape(-1, env.observation_space.shape[0])

    Z = np.zeros(XY_combos.shape[0])
    A = np.zeros(XY_combos.shape[0], dtype=int)
    for Idx, state in enumerate(XY_combos):
        values = predict(state.reshape(1,-1))
        A[Idx] = np.argmax(values)
        Z[Idx] = values[A[Idx]]
    Z = Z.reshape(X.shape).T
    A = A.reshape(X.shape).T

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

def TestPolicy(env, policy, b_render):
    rewards = 0
    observation = env.reset()
    while True:
        if b_render: env.render()
        action = policy(observation.reshape(1, -1))
        observation_new, reward, done, info = env.step(action)
        rewards += reward
        observation = observation_new
        if done: break
    return rewards
def SampleGenerator(env, samplemethod, samplerange, observe_scalars=1, true_scalars=1):
    
    env.reset()
    true_state_length = len(env.env.state)
    observable_state_length = env.observation_space.shape[0]

    observed_samples = np.empty(shape=(samplerange, observable_state_length))
    if samplemethod is 'PolicySampledExperience':
        for ix, (low, high) in enumerate(zip(env.observation_space.low,env.observation_space.high)):
            observed_samples[:,ix] = np.random.uniform(low=low, high=high, size=samplerange)
        true_samples = observed_samples
    elif samplemethod is 'PolicyRandomExperience':
        true_samples = np.empty(shape=(samplerange, true_state_length))
        for ix in range(samplerange):
            observed_samples[ix,:] = env.reset()*observe_scalars
            true_samples[ix,:] = env.env.state*true_scalars
    
    return observed_samples, true_samples

def LSPI(env, estimator, discount_factor=0.99, dynamic_experience = False):

    reward_end = 0
    samplerange = 50000
    
     # Training data changes throughout
    observed_samples, true_samples = SampleGenerator(env, samplemethod, samplerange)
    experience = RandomExperience(env, estimator, samples, true_samples)

    for episode in range(100):
        
        estimator.update_LSQ(experience, discount_factor, episode,
            dynamic_experience=dynamic_experience)
        
        if (episode%1)==0: PlotValue(env,estimator, episode)

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


def Deep_D_AC_OffP_PG(env, actor_critic, num_episodes, samplemethod,
    discount_factor=0.99, n_mini_batch=100, visual_updates=10, plot_updates=10,
    observe_scalars=1, true_scalars=1, save_updates=25, BRender=False):
    
    mini_batch = actor_critic.mini_batch
    for episode in range(num_episodes):
   
        if (episode%plot_updates == 0) & (episode>0) & BRender:
            if env.observation_space.shape[0] == 2:
                PlotValue(env, actor_critic.plot_policy_value, episode)
                PlotAction(env, actor_critic.predict_actor, episode)
            else:
                PlotPC(env, [actor_critic.plot_policy_value, actor_critic.predict_actor], actor_critic.state_labels, episode, observe_scalars, ['Value', 'Policy'])
        
        generator_size = n_mini_batch * mini_batch

        observed_samples, true_samples = SampleGenerator(env, samplemethod, generator_size, 
                observe_scalars=observe_scalars, true_scalars=true_scalars)
        batch_observed_samples = np.split(observed_samples, n_mini_batch)
        batch_true_samples = np.split(true_samples, n_mini_batch)

        #with click.progressbar(range(n_mini_batch)) as bar:
        #for idx in bar:
        t_samples = 0
        t_predict = 0
        t_temporal = 0
        t_traincritic = 0
        t_trainactor = 0
        t_traintarget = 0

        for idx in range(n_mini_batch):
            
            start = timer()

            sampled_experience = PolicyExperienceBatch(
                env, actor_critic.predict_actor, batch_observed_samples[idx], batch_true_samples[idx], policy_std=1)     
            observations, actions, rewards, observations_new, dones = map(
                np.squeeze,(map(np.array,zip(*sampled_experience))))
            end = timer()
            t_samples += end-start
            start = end

            target_policy_actions_new = actor_critic.predict_target_actor(observations_new)
            policy_actions = actor_critic.predict_actor(observations)
            values_new = actor_critic.predict_target_critic(observations_new, target_policy_actions_new)
            end = timer()
            t_predict += end-start
            start = end

            undones = [not i for i in dones]
            idx_dones = np.where(dones)[0]
            idx_undones = np.where(undones)[0]
            td_targets = np.zeros(mini_batch)
            td_targets[idx_dones] = rewards[idx_dones].flatten()
            td_targets[idx_undones] = rewards[idx_undones].flatten() + discount_factor * values_new[idx_undones].flatten()
            end = timer()
            t_temporal += end-start
            start = end
            
            actor_critic.train_critic(observations, actions.reshape(-1, 1), td_targets.reshape(-1, 1))
            end = timer()
            t_traincritic += end-start
            start = end

            critic_gradients = actor_critic.critic_gradients(observations, policy_actions)[0]
            actor_critic.train_actor(observations, critic_gradients)
            end = timer()
            t_trainactor += end-start
            start = end

            actor_critic.train_target()
            end = timer()
            t_traintarget += end-start

        times = [t_samples, t_predict, t_temporal, t_traincritic, t_trainactor, t_traintarget]
        print('Samples, Predict, TD, Critic, Actor, Targets')
        print(["%.4f" % (i/n_mini_batch) for i in times])

        if (episode%visual_updates == 0) & BRender: b_render = True
        else: b_render = False
        rewards = TestPolicy(env, actor_critic.predict_actor, b_render)
        print('  ' + str(episode) + ' - ' + str(rewards))

def Deep_D_AC_OffP_PG_exp(env, actor_critic, num_episodes,
    discount_factor=0.99, visual_updates=10, plot_updates=10, save_updates=25,
    observe_scalars=1, true_scalars=1, random_updates=10, BRender=False):
    
    actor_critic.train_target()

    mini_batch = actor_critic.mini_batch

    experience = deque(maxlen=10000)
    with click.progressbar(range(num_episodes)) as bar:
        for episode in bar:

            if (episode%plot_updates == 0) & (episode>0):
                if env.observation_space.shape[0] == 2:
                    PlotValue(env, actor_critic.plot_policy_value, episode)
                    PlotAction(env, actor_critic.predict_actor, episode)
                else:
                    PlotPC(env, [actor_critic.plot_policy_value, actor_critic.predict_actor],
                         actor_critic.state_labels, episode, observe_scalars, ['Value', 'Policy'])
        
            observation = env.reset().reshape(1, -1)
            reward_tracker = 0
            while True:

                if (episode%visual_updates)==0 & BRender: env.render()
                '''
                action_mean = actor_critic.predict_actor(observation)
                if (episode%random_updates)==0: action = action_mean
                else: action = action_mean + 1/(1+np.float(episode)) #action = np.random.normal(loc=action_mean, scale=1)
                '''
                action = actor_critic.predict_actor(observation)+ 1/(1+np.float(episode))
                observation_new, reward, done, info = env.step(action[0]) #NOTE: added a [0]
                reward_tracker += reward
                experience.append((observation, action, reward, observation_new, done))
                observation = observation_new.reshape(1, -1)
                
                if len(experience) > mini_batch:
                    random_ix = np.random.randint(low=0, high=len(experience), size=mini_batch)
                    batch = itemgetter(*random_ix)(experience)
                    observations, actions, rewards, observations_new, dones = map(
                        np.squeeze,(map(np.array,zip(*batch))))
                    
                    target_policy_actions_new = actor_critic.predict_target_actor(observations_new)
                    values_new = actor_critic.predict_target_critic(observations_new, target_policy_actions_new)

                    undones = [not i for i in dones]
                    idx_dones = np.where(dones)[0]
                    idx_undones = np.where(undones)[0]
                    td_targets = np.zeros(mini_batch)
                    td_targets[idx_dones] = rewards[idx_dones].flatten()
                    td_targets[idx_undones] = rewards[idx_undones] + discount_factor * values_new[idx_undones].flatten()
                    actor_critic.train_critic(observations, actions.reshape(-1, 1), td_targets.reshape(-1, 1))
                
                    policy_actions = actor_critic.predict_actor(observations)
                    critic_gradients = actor_critic.critic_gradients(observations, policy_actions)[0]
                    actor_critic.train_actor(observations, critic_gradients)
                    actor_critic.train_target()

                    '''
                    target_policy_actions_new = actor_critic.predict_actor(observations_new, actor_critic.actor_target)
                    policy_actions = actor_critic.predict_actor(observations, actor_critic.actor)
                    values_new = actor_critic.critic_target.predict([observations_new[idx_undones], target_policy_actions_new[idx_undones]])
                    actor_critic.train_target(actor_critic.critic, actor_critic.critic_target, actor_critic.tau)
                    actor_critic.train_target(actor_critic.actor, actor_critic.actor_target, actor_critic.tau)
                    '''
                
                if done: 
                    print('  ' + str(episode) + ' - ' + str(reward_tracker))
                    break

def D_AC_OffP_PG(env, policy_estimator, value_estimator, advantage_estimator, num_episodes, samplemethod, samplerange=1000000,
    discount_factor=0.99, alpha_w=1e-3, alpha_v=1e-3, alpha_theta=1e-4, batch_size=1000, visual_updates=10, plot_updates=10,
    observe_scalars=1, true_scalars=1, save_updates=25):
    '''
    Deterministic Off Policy Actor Critic Policy Gradient
    '''

    for episode in range(num_episodes):

        if (episode%save_updates == 0):
            pickle.dump(advantage_estimator.weights, open("savedadvantagemodel.txt", "wb" ) )
            pickle.dump(value_estimator.weights, open("savedvaluemodel.txt", "wb" ) )
            pickle.dump(policy_estimator.weights, open("savedpolicymodel.txt", "wb" ) )

        if (episode%plot_updates == 0):
            if value_estimator.state_length == 2:
                PlotValue(env, value_estimator, episode)
                PlotAction(env, policy_estimator, episode)
            else:
                PlotPC(env, [value_estimator, policy_estimator], policy_estimator.state_labels, episode, observe_scalars, ['Value', 'Policy'])

        observed_samples, true_samples = SampleGenerator(env, samplemethod, batch_size, 
            observe_scalars=observe_scalars, true_scalars=true_scalars)
        sampled_experience = PolicyExperience(env, policy_estimator.predict, observed_samples, true_samples, policy_std=1)

        with click.progressbar(range(len(sampled_experience))) as bar:
            for idx in bar:
                (observation, action, reward, observation_new, done) = sampled_experience[idx]
                
                policy_action = policy_estimator.predict(observation, policy_estimator.weights)
                policy_feature, policy_feature_ix = policy_estimator.featurize_state(observation)

                value = value_estimator.predict(observation, value_estimator.weights)
                value_feature, _ = value_estimator.featurize_state(observation)
                value_new = value_estimator.predict(observation_new, value_estimator.weights)

                advantage_feature = advantage_estimator.featurize_state(observation, policy_feature, action, policy_action)
                advantage = advantage_estimator.predict(advantage_feature, advantage_estimator.weights)

                state_action_value = value + advantage
                state_action_value_new = value_new # state_action_value_new is purely the value, as the action chosen is the policy
                
                if done:
                    td_target = reward
                else:
                    td_target = reward + discount_factor * state_action_value_new

                td_error = td_target - state_action_value
                # Can speed this up for tile coding
                # policy_error = np.inner(policy_feature.flatten(), advantage_estimator.weights.flatten())
                policy_error = np.sum(advantage_estimator.weights[policy_feature_ix])
                
                advantage_estimator.update_SGD(advantage_feature, td_error, alpha_w)
                value_estimator.update_SGD(value_feature, td_error, alpha_v)
                policy_estimator.update_SGD(policy_feature, policy_error, alpha_theta)
                
        
        rewards = 0
        observation = env.reset()
        while True:
            if (episode%visual_updates == 0): env.render()
            action = policy_estimator.predict(observation, policy_estimator.weights)
            observation_new, reward, done, info = env.step(action)
            rewards += reward
            observation = observation_new
            if done: 
                print(str(episode) + ' - ' + str(rewards))
                break
    
    return policy_estimator, value_estimator, advantage_estimator

def Stoc_AC_OnP_PG(env, policy_estimator, value_estimator, num_episodes, display=True,
        epsilon_decay=0.99, epsilon_initial=0.25, visual_updates=1, policy_std=10**-1,
        value_alpha=1e-3,  value_discount_factor=0.99,  value_llambda=0, value_momentum=0,
        policy_alpha=1e-3, policy_discount_factor=0.99, policy_llambda=0, policy_momentum=0,
        samplerange=5000, samplemethod='PolicySampledExperience', plot_updates=1,
        sample_scalars=1, dynamic_experience = False):
    '''
    Stochastic On Policy Actor Critic Policy Gradient
    '''

    observed_samples, true_samples = SampleGenerator(env, samplemethod, samplerange, sample_scalars)
    
    for episode in range(num_episodes):
   
        experience = PolicyExperience(env, policy_estimator.predict, 
            observed_samples, true_samples, policy_std=policy_std) 

        value_estimator.update_LSTD(env,
            experience, value_discount_factor, dynamic_experience=dynamic_experience)
        
        if (episode%plot_updates == 0):
            
            if value_estimator.state_length == 2:
                PlotValue(env, value_estimator, episode)
                PlotAction(env, policy_estimator, episode)
            else:
                PlotPC(env, [value_estimator, policy_estimator], value_estimator.state_labels, episode, sample_scalars, ['Value', 'Policy'])

        episode_tracker = []
        value_eligibility = np.zeros(value_estimator.feature_length)
        policy_eligibility = np.zeros(policy_estimator.feature_length)
        value_update = 0
        policy_update = 0
        observation = env.reset()
        rewards = 0
        steps = 0
        while True:
            steps += 1

            if (episode%visual_updates == 0) & display: env.render()

            mean_action, action, policy_score = policy_estimator.predict_Gaussian(observation, policy_estimator.weights, policy_std)
            observation_new, reward, done, info = env.step(action)
            rewards += reward

            # Value TD error
            value = value_estimator.predict(observation, value_estimator.weights)
            value_feature, _ = value_estimator.featurize_state(observation)
            value_new = value_estimator.predict(observation_new, value_estimator.weights)

            if done & (env._elapsed_steps<env._max_episode_steps): #SYSTEM CAN SAY DONE JUST DUE TO TIMEOUT!
                value_td_target = reward
            else:
                value_td_target = reward + value_discount_factor * value_new
            
            value_td_error = value_td_target - value

            if abs(value_td_error)>1000:
                print('Resetting due to high TD error')
                break
            
            # print(str(value_td_error) + '-' + str(reward) + '-' + str(action))
            
            # Eligibility decay
            value_eligibility = value_discount_factor * value_llambda * value_eligibility
            value_eligibility += value_feature.T.flatten()/value_estimator.NTiling
            policy_eligibility = policy_llambda * policy_eligibility
            policy_eligibility += policy_score.flatten()
            
            # Update weights                    
            '''
            value_update = value_estimator.update(
                value_eligibility, value_td_error, value_alpha, value_update, value_momentum)
            '''
            policy_update = policy_estimator.update_SGD(
                policy_eligibility, policy_alpha, value_td_error, policy_update, policy_momentum)
            
            if done:
                print(str(episode) + ' - ' + str(rewards))
                break

            # Loop back parameters
            observation = observation_new
    
    return value_estimator, policy_estimator

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
                update = estimator.update_SGD(eligibility, td_error, alpha, update, momentum)

                observation = observation_new

                # Update statistics
                stats.episode_rewards[episode] += reward  
                
                if done:
                    stats.episode_lengths[episode] = counter
                    break
    
    return stats, estimator