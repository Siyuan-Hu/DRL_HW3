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


class A2C(object):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self,
                 env,
                 model,
                 lr,
                 critic_model,
                 critic_lr,
                 num_episodes,
                 N_step=20,
                 render=False,
                 discount_factor=0.01):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - N_step: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.N_step = N_step

        # enviroment
        self.num_action = env.action_space.n
        self.num_observation = env.observation_space.shape[0]
        self.num_episodes = num_episodes
        self.render = render
        self.discount_factor = discount_factor

        # model
        self.lr = lr
        self.critic_lr = critic_lr

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

        # define the network for the actor
        self.G = tf.placeholder(tf.float32,
                           shape=[None],
                           name='G')
        self.action = tf.placeholder(tf.float32,
                                shape=[None, self.num_action],
                                name='action')
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.output = self.model.output
        score_func = tf.reduce_sum(tf.multiply(self.output, self.action), axis=[1], keepdims=True)
        score_func = tf.log(score_func)
        loss = - tf.reduce_mean(tf.multiply(self.G, score_func), axis=0)
        gradient = optimizer.compute_gradients(loss, model.weights)
        self.updata_weights = optimizer.apply_gradients(gradient)


        # define the network for the critic
        self.critic_state_input = tf.placeholder(tf.float32,
                                                 shape=[None, self.num_observation],
                                                 name="critic_state_input")
        self.critic_action_input = tf.placeholder(tf.float32,
                                                  shape=[None, self.num_action],
                                                  name="critic_action_input")
        self.target_q_value = tf.placeholder(tf.float32,
                                             shape=[None],
                                             name="target_q_value")

        ## TODO
        # create network
        # get critic output
        critic_output = self.critic_model.output
        q_value_output = tf.reduce_sum(tf.multiply(critic_output, self.critic_action_input), axis=[1])
        critic_loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, q_value_output)))
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(critic_loss, name = "critic_optimizer")
        
        ## TODO
        # whether one or two session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        test_frequence = 200
        test_total_reward = 0
        for i in range(self.num_episodes):
            states, actions, rewards = self.generate_episode(sess=self.sess,
                                                             model=self.model,
                                                             env=env,
                                                             render=self.render)
            R, G = self.episode_reward2G_Nstep(critic_sess=self.critic_sess,
                                            states=states,
                                            actions=actions,
                                            rewards=rewards,
                                            gamma=gamma,
                                            N_step=self.N_step,
                                            discount_factor=self.discount_factor)
            self.sess.run(self.updata_weights,
                          feed_dict={self.model.input: states,
                                     self.G: G,
                                     self.action: actions})

            ## TODO
            # how to run the critic session and update the critic network
            self.critic_optimizer.run(sess=self.critic_sess,
                                      feed_dict={self.critic_state_input: states,
                                                 self.critic_action_input: actions,
                                                 self.target_q_value: R})

        return

    def generate_episode(self, sess, model, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        state = env.reset()
        num_observation = env.observation_space.shape[0]
        while True:
            if render:
                env.render()
            action = self.output.eval(session = sess, feed_dict={model.input: [state]})
            one_hot_action = Reinforce.get_random_one_hot_action(action)
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

    def episode_reward2G_Nstep(self, critic_sess, states, actions, rewards, gamma, N_step, discount_factor):
        ## TODO
        # how to get the output
        critic_output = self.critic_model.eval(session=critic_sess,
                                               feed_dict={self.critic_state_input: states})
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
            G[t] = R[t] - critic_output[t]

        return R, G

    @staticmethod
    def sum_rewards(rewards):
        return sum(rewards)

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
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
