import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

ENVIROMENT = 'InvertedPendulum-v1'
LAYER1_SIZE=400
LAYER2_SIZE=300
class A2C(object):
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	def __init__(self,
				 env,
				 model_config_path,
				 actor_lr,
				 critic_lr,
				 num_episodes,
				 N_step=0,
				 render=False,
				 discount_factor=0.99,
				 model_step = None):
		# Initializes A2C.
		# Args:
		# - model: The actor model.
		# - lr: Learning rate for the actor model.
		# - critic_model: The critic model.
		# - critic_lr: Learning rate for the critic model.
		# - N_step: The value of N in N-step A2C.
		self.env = env
		self.N_step = N_step
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		# enviroment
		num_action = env.action_space.shape[0]
		num_observation = env.observation_space.shape[0]
		self.num_episodes = num_episodes
		self.render = render
		self.discount_factor = discount_factor
		self.sess = tf.Session()
		# model
		self.actor_model = Actor(self.sess,num_observation,num_action,self.actor_lr)
		self.critic_model = Critic(self.sess,num_observation,num_action ,self.critic_lr)

	

	def train(self, gamma=1.0):
		# Trains the model on a single episode using A2C.
		file = open("log.txt", "w")

		test_frequence = 1000
		self.gamma_N_step = gamma ** self.N_step
		for i in range(self.num_episodes):
			states, actions, rewards = self.generate_episode(env=self.env,
															 render=self.render)
			R, G = self.episode_reward2G_Nstep(states=states, actions=actions, rewards=rewards, 
				gamma=gamma, N_step=self.N_step, discount_factor=self.discount_factor)
			self.critic_model.train(states, R,actions)
			action_batch_for_gradients = self.actor_model.actions(states)
			q_gradient_batch = self.critic_model.gradients(states,action_batch_for_gradients)
			self.actor_model.train(q_gradient_batch,states)
			

			if (i % test_frequence == 0):
				reward, std = self.test(i)
				file.write(str(reward)+" "+str(std)+"\n")
				print(reward, std)


		file.close()

	def test(self, epc_idx):
		log_dir = './log'
		name_mean = 'test10_reward'
		name_std = 'test10_std'
		num_test = 100
		total_array = np.zeros(num_test)
		env = gym.make(ENVIROMENT)
		for j in range(num_test):
			_, _, rs = self.generate_episode(env)
			total_array[j] = A2C.sum_rewards(rs)

		summary_var(log_dir, name_mean, np.mean(total_array), epc_idx)
		summary_var(log_dir, name_std, np.std(total_array), epc_idx)
		env.close()

		return np.mean(total_array), np.std(total_array)

	def get_action(self,state):
		action = self.actor_model.action(state)
		return action

	def generate_episode(self, env, render=False):
		# Generates an episode by running the given model on the given env.
		# Returns:
		# - a list of states, indexed by time step
		# - a list of actions, indexed by time step
		# - a list of rewards, indexed by time step
		states = []
		actions = []
		rewards = []

		state = env.reset()
		num_observation = env.observation_space.shape[0]
		while True:
			if render:
				env.render()
			action = self.actor_model.action(state)


			states.append(state)
			actions.append(action)
			state, reward, done, info = env.step(action)
			rewards.append(reward)
			if done:
				break

		return states, actions, rewards


	def episode_reward2G_Nstep(self, states, actions, rewards, gamma, N_step, discount_factor):
		## TODO
		# how to get the output
		critic_output = self.critic_model.get_critics(states,actions)
		num_total_step = len(rewards)
		# R: list, is the symbol "R_t" in the alorithm 2
		# G: list, is the difference between R and V(S_t)
		R = [None] * num_total_step
		G = [None] * num_total_step
		for t in range(num_total_step - 1, -1, -1):
			V_end = 0 if (t + N_step >= num_total_step) else critic_output[t + N_step]
			R[t] = (self.gamma_N_step) * V_end
			gamma_k = 1
			for k in range(N_step):
				R[t] += discount_factor * (gamma_k) * (rewards[t + k] if (t + k < num_total_step) else 0)
				gamma_k *= k
			G[t] = R[t] - critic_output[t][0]

		return R, G

	@staticmethod
	def sum_rewards(rewards):
		return sum(rewards)

	def save_models(self, model_step):
		self.actor_model.save_model(model_step)
		self.critic_model.save_model(model_step)

	def load_models(self, model_config_path, num_observation, num_action, actor_lr, critic_lr, model_step):
		self.actor_model = Actor(model_config_path, num_action, actor_lr, model = "./models/actor-"+str(model_step)+".h5")
		self.critic_model = Critic(num_observation, critic_lr, model = "./models/critic-"+str(model_step))

def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-config-path', dest='model_config_path',
						type=str, default='LunarLander-v2-config.json',
						help="Path to the actor model config file.")
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=5000000, help="Number of episodes to train on.")
	parser.add_argument('--lr', dest='lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--N_step', dest='N_step', type=int,
						default=20, help="The value of N in N-step A2C.")

	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()

class Actor(object):

	def __init__(self,sess,state_dim,action_dim,actor_lr):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.actor_lr = actor_lr
		# create actor network
		self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

		# define training rules
		self.create_training_method()

		self.sess.run(tf.initialize_all_variables())

		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(self.parameters_gradients,self.net))

	def create_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
		action_output = tf.tanh(tf.matmul(layer2,W3) + b3)

		return state_input,action_output,[W1,b1,W2,b2,W3,b3]


	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]


	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def save_model(self, step):
		# Helper function to save your model.
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = step)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(model_file)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path

class Critic(object):
	def __init__(self, sess,num_observation, num_action,lr):
		# define the network for the critic
		self.num_observation = num_observation
		self.num_action = num_action
		self.learning_rate = lr

		self.sess = tf.Session()


		self.state_input,self.action_input,self.q_value_output,self.net = self.create_q_network(num_observation,num_action)
		self.create_optimizer()
		self.sess.run(tf.global_variables_initializer())



	def create_q_network(self,state_dim,action_dim):
		# the layer size could be changed
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
		W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
		b2 = self.variable([layer2_size],layer1_size+action_dim)
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)

		return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]


	def create_optimizer(self):
		# Using Adam to minimize the error between target and evaluation
		self.target_q_value = tf.placeholder(tf.float32, [None], name = "target_q_value")
		cost = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, self.q_value_output)))
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name = "optimizer")
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def train(self, states, R,actions):
		self.sess.run(self.optimizer,feed_dict={self.target_q_value:R,self.state_input:states,self.action_input:actions})
		#self.optimizer.run(session=self.sess, feed_dict={self.state_input: states, self.target_q_value: R})

	def gradients(self,states,actions):
		return self.sess.run(self.action_gradients,feed_dict={self.state_input:states,self.action_input:actions})[0]

	def get_critics(self, states,actions):
		return self.sess.run(self.q_value_output,feed_dict={self.state_input:states,self.action_input:actions})
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def save_model(self, step):
		# Helper function to save your model.
		saver = tf.train.Saver()
		self.sess.graph.add_to_collection("optimizer", self.optimizer)
		saver.save(self.sess, "./models/critic", global_step = step)
	'''
	def load_model(self, model_file):
		# Helper function to load an existing model.
		tf.reset_default_graph()
		self.sess = tf.Session()
		saver = tf.train.import_meta_graph(model_file + '.meta')
		saver.restore(self.sess, model_file)

		graph = tf.get_default_graph()
		self.q_values = graph.get_tensor_by_name("q_values:0")
		self.state_input = graph.get_tensor_by_name("state_input:0")
		self.target_q_value = graph.get_tensor_by_name("target_q_value:0")
		self.optimizer = graph.get_collection("optimizer")[0]
	'''
from tensorflow.core.framework import summary_pb2
def summary_var(log_dir, name, val, step):
	writer = tf.summary.FileWriterCache.get(log_dir)
	summary_proto = summary_pb2.Summary()
	value = summary_proto.value.add()
	value.tag = name
	value.simple_value = float(val)
	writer.add_summary(summary_proto, step)
	writer.flush()

def main(args):
	# Parse command-line arguments.
	args = parse_arguments()
	model_config_path = args.model_config_path
	num_episodes = args.num_episodes
	lr = args.lr
	critic_lr = args.critic_lr
	N_step = args.N_step
	render = args.render

	# Create the environment.
	env = gym.make(ENVIROMENT)
	
	a2c = A2C(env, model_config_path, lr, critic_lr, num_episodes, N_step, render)#, model_step = 7400)

	a2c.train()

	# TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
	main(sys.argv)
