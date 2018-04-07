import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

ENVIROMENT = 'LunarLander-v2'

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
                 N_step=20,
                 render=False,
                 discount_factor=0.01,
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

        # enviroment
        num_action = env.action_space.n
        num_observation = env.observation_space.shape[0]
        self.num_episodes = num_episodes
        self.render = render
        self.discount_factor = discount_factor

        # model
        if model_step == None:
            self.actor_model = Actor(model_config_path, num_action, actor_lr)
            self.critic_model = Critic(num_observation, critic_lr)
        else:
            self.load_models(model_config_path, num_observation, num_action, actor_lr, critic_lr, model_step)


    def train(self, gamma=1.0):
        # Trains the model on a single episode using A2C.
        max_reward = -500
        test_frequence = 200
        for i in range(self.num_episodes):
            states, actions, rewards = self.generate_episode(env=self.env,
                                                             render=self.render)
            R, G = self.episode_reward2G_Nstep(states=states, actions=actions, rewards=rewards, 
                gamma=gamma, N_step=self.N_step, discount_factor=self.discount_factor)
            self.actor_model.train(states, G, actions)
            self.critic_model.train(states, R)

            if (i % test_frequence == 0):
                reward = self.test(i)
                print(reward)
                if reward > max_reward:
                    self.save_models(i)
                    max_reward = reward

    def test(self, epc_idx):
        log_dir = './log'
        name_mean = 'test10_reward'
        name_std = 'test10_std'
        num_test = 10
        total_array = np.zeros(num_test)
        env = gym.make(ENVIROMENT)
        for j in range(10):
            _, _, rs = self.generate_episode(env)
            total_array[j] = A2C.sum_rewards(rs)

        summary_var(log_dir, name_mean, np.mean(total_array), epc_idx)
        summary_var(log_dir, name_std, np.std(total_array), epc_idx)
        env.close()

        return np.sum(total_array) / num_test

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
            action = self.actor_model.get_action(state)
            one_hot_action = A2C.get_random_one_hot_action(action)
            states.append(state)
            actions.append(one_hot_action)
            state, reward, done, info = env.step(np.argmax(one_hot_action))
            rewards.append(reward)
            if done:
                break

        return states, actions, rewards

    @staticmethod
    def get_random_one_hot_action(action):
        random_number = np.random.rand()
        num_action = action.shape[1]
        prob_sum = 0
        one_hot_action = np.zeros(num_action)
        for i in range(num_action):
            prob_sum += action[0, i]
            if random_number <= prob_sum:
                one_hot_action[i] = 1
                break
        return one_hot_action

    def episode_reward2G_Nstep(self, states, actions, rewards, gamma, N_step, discount_factor):
        ## TODO
        # how to get the output
        critic_output = self.critic_model.get_critics(states)
        num_total_step = len(rewards)
        # R: list, is the symbol "R_t" in the alorithm 2
        # G: list, is the difference between R and V(S_t)
        R = [None] * num_total_step
        G = [None] * num_total_step
        for t in range(num_total_step - 1, -1, -1):
            V_end = 0 if (t + N_step >= num_total_step) else critic_output[t + N_step]
            R[t] = (gamma ** N_step) * V_end
            for k in range(N_step):
                R[t] += discount_factor * (gamma ** k) * (rewards[t + k] if (t + k < num_total_step) else 0)
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
                        default=50000, help="Number of episodes to train on.")
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
    def __init__(self, model_config_path, num_action, lr, model = None):
        # if model != None:
        #     self.load_model(model)
        #     return

        with open(model_config_path, 'r') as f:
            self.model = keras.models.model_from_json(f.read())


        # define the network for the actor
        self.G = tf.placeholder(tf.float32,
                           shape=[None],
                           name='G')
        self.action = tf.placeholder(tf.float32,
                                shape=[None, num_action],
                                name='action')
        self.input = self.model.input
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.output = self.model.output
        score_func = tf.reduce_sum(tf.multiply(self.output, self.action), axis=[1], keepdims=True)
        score_func = tf.log(score_func)
        loss = - tf.reduce_mean(tf.multiply(self.G, score_func), axis=0)
        gradient = optimizer.compute_gradients(loss, self.model.weights)
        self.updata_weights = optimizer.apply_gradients(gradient)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        keras.backend.set_session(self.sess)

        if model != None:
            self.load_model(model)
            print("load")
        # self.sess.run(tf.local_variables_initializer())
        # print("hehe")

    def train(self, states, G, actions):
        self.sess.run(self.updata_weights, 
            feed_dict={self.input: states, self.G: G, self.action: actions})

    def get_action(self, state):
        # action = self.output.eval(session = self.sess, feed_dict={self.input: [state]})
        # # print("~~~~~~~~")
        # print(action)
        # print(self.model.predict_on_batch(state.reshape(1,8)))
        return self.output.eval(session = self.sess, feed_dict={self.input: [state]})
        # return self.model.predict_on_batch(state.reshape(1,8))

    # def save_model_weight(self, step):
    #     self.model.save_weights("./models/actor-"+str(step)+".h5")

    # def load_model_weight(self, weight_file):
    #     self.model.load_weights(weight_file)

    def save_model(self, step):
        # Helper function to save your model.
        self.model.save_weights("./models/actor-"+str(step)+".h5")

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model.load_weights(model_file)

class Critic(object):
    def __init__(self, num_observation, lr, model = None):
        # define the network for the critic
        self.num_observation = num_observation
        self.learning_rate = lr

        self.sess = tf.Session()

        if model != None:
            self.load_model(model)
        else:
            self.create_mlp()
            self.create_optimizer()
            self.sess.run(tf.global_variables_initializer())

    def train(self, states, R):
        self.optimizer.run(session=self.sess, feed_dict={self.state_input: states, self.target_q_value: R})

    def create_mlp(self):
        # Craete multilayer perceptron (one hidden layer with 20 units)
        self.hidden_units = 20

        self.w1 = self.create_weights([self.num_observation, self.hidden_units])
        self.b1 = self.create_bias([self.hidden_units])

        self.state_input = tf.placeholder(tf.float32, [None, self.num_observation], name = "state_input")

        h_layer = tf.nn.relu(tf.matmul(self.state_input, self.w1) + self.b1)

        self.w2 = self.create_weights([self.hidden_units, 1])
        self.b2 = self.create_bias([1])
        self.q_values = tf.add(tf.matmul(h_layer, self.w2), self.b2, name = "q_values")

    def create_weights(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def create_bias(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def create_optimizer(self):
        # Using Adam to minimize the error between target and evaluation
        self.target_q_value = tf.placeholder(tf.float32, [None], name = "target_q_value")
        cost = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, self.q_values)))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name = "optimizer")

    def get_critics(self, states):
        return self.q_values.eval(session = self.sess, feed_dict={self.state_input: states})

    def save_model(self, step):
        # Helper function to save your model.
        saver = tf.train.Saver()
        self.sess.graph.add_to_collection("optimizer", self.optimizer)
        saver.save(self.sess, "./models/critic", global_step = step)

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
    
    a2c = A2C(env, model_config_path, lr, critic_lr, num_episodes, N_step, render, model_step = 7400)

    a2c.train()

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
