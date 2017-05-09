import matplotlib
import gym
import pickle

import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from lib import plotting
from lib import EpsilonFunction
from lib.RL_Library import NN_DDPG_Estimator, Deep_D_AC_OffP_PG, Deep_D_AC_OffP_PG_exp

visual_updates = 10
mini_batch = 64

test = 'Pendulum-v0'

if test is "MountainCarContinuous-v0":
    method = "PolicySampledExperience"
    state_labels = ['x', 'x_dot']
    observe_scalars = 1
    true_scalars = 1
    actor_alpha = 0.0001
    critic_alpha = 0.001

elif test is "Pendulum-v0":
    method = "PolicyRandomExperience"
    state_labels = ['cos(θ)', 'sin(θ)', 'θ_dot']
    observe_scalars = [1, 1, 8]
    true_scalars = [1, 8]
    actor_alpha = 0.0001
    critic_alpha = 0.001

sess = tf.Session()
env = gym.envs.make(test)

algorithm = 1

'''
Can we use an expected reward range?
'''

weights = None
num_episodes = 10000

if algorithm==1:
    n_mini_batch = 1000
    plot_updates = 25

    actor_critic = NN_DDPG_Estimator(env, sess, state_labels, 
                                    actor_alpha=actor_alpha, critic_alpha=critic_alpha,
                                    actor_NLayers=[400,300], critic_NLayers=[400,300],
                                    mini_batch=mini_batch, tau=0.001)

    Deep_D_AC_OffP_PG(env, actor_critic, num_episodes, method,
                                    plot_updates=plot_updates, visual_updates=visual_updates,
                                    observe_scalars=observe_scalars, true_scalars=true_scalars)

elif algorithm == 2:

    plot_updates = 100
    actor_critic = NN_DDPG_Estimator(env, sess, state_labels, 
                                     actor_alpha=actor_alpha, critic_alpha=critic_alpha,
                                     actor_NLayers=[400,300], critic_NLayers=[400,300],
                                     mini_batch=mini_batch, tau=0.001)

    Deep_D_AC_OffP_PG_exp(env, actor_critic, num_episodes,
                          visual_updates=visual_updates, plot_updates=plot_updates,
                          observe_scalars=observe_scalars, true_scalars=true_scalars)


'''
Normalise inputs to have min/max of -1 1
'''

'''
finally understand the gradient function.

chain rule

dcritic/dactions; how much does action have to change to achieve ~learning rate~ change in value
dactor/dweights; how much do weights have to change to achieve corresponding change in action

'''

'''
values = self.critic.predict([states,actions]) # At the state AND action
action_policy = self.actor.predict(states)
values_policy = self.critic.predict([states,action_policy]) # At the state, following policy action

critic_gradients = self.sess.run(self.critic_grads, feed_dict={
    self.states: states,
    self.actions: actions})
#action_test = actions - self.actor_alpha/critic_gradients[0] # --- shows that we can achieve the step change in value
actions_test = actions + self.actor_alpha*critic_gradients[0]
values_test = self.critic.predict([states, actions_test])
delta_check = values_test - values

actor_gradients = self.sess.run(self.actor_grads, feed_dict={
    self.states: states,
    self.actions: actions})

weights = self.actor.get_weights()
weights_new = self.actor.get_weights()

for i in range(len(weights)):
    weights_new[i] = weights[i] - actor_gradients[i] * self.actor_alpha

self.actor.set_weights(weights_new)

actions_new = self.actor.predict(states)
values_new = self.critic.predict([states, actions_new])

### RESET
self.actor.set_weights(weights)

###

self.sess.run(self.actor_optimize, feed_dict={
    self.states: states,
    self.actions: actions
})

actions_new2 = self.actor.predict(states)
values_new2 = self.critic.predict([states, actions_new])

'''