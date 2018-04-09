import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self,
                 env,
                 model,
                 learning_rate,
                 num_episodes,
                 render=False):
        self.model = model
        ## TODO 
        # enviroment
        self.num_action = env.action_space.n
        self.num_observation = env.observation_space.shape[0]
        self.num_episodes = num_episodes
        self.render = render

        # model
        self.learning_rate = learning_rate

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.G = tf.placeholder(tf.float32,
                           shape=[None],
                           name='G')
        self.action = tf.placeholder(tf.float32,
                                shape=[None, self.num_action],
                                name='action')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # output = tf.squeeze(self.model.output)
        self.output = self.model.output
        score_func = tf.reduce_sum(tf.multiply(self.output, self.action), axis=[1],keepdims=True)
        score_func = tf.log(score_func)
        ##TODO 
        # divide size
        # loss = - tf.tensordot(self.G, score_func, axes=0)
        loss = - tf.reduce_mean(tf.multiply(self.G, score_func), axis=0)
        gradient = optimizer.compute_gradients(loss, model.weights)
        self.updata_weights = optimizer.apply_gradients(gradient)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        file = open("log.txt", "w")

        frequence = 3000
        num_test = 100
        total_array = np.zeros(num_test)
        
        for i in range(self.num_episodes):
            states, actions, rewards = self.generate_episode(self.sess, self.model, env)
            
            G = Reinforce.episode_reward2G(rewards, gamma)

            self.sess.run(self.updata_weights,
                     feed_dict={self.model.input: states,
                                self.G: G,
                                self.action: actions})

            if (i % frequence == 0):
                _env = gym.make('LunarLander-v2')
                for j in range(num_test):
                    _, _, rs = self.generate_episode(self.sess, self.model, _env)
                    tmp_1 = Reinforce.sum_rewards(rs)
                    total_array[j] = tmp_1
                test_reward_mean = np.mean(total_array)
                test_reward_std = np.std(total_array)

                file.write(str(test_reward_mean)+" "+str(test_reward_std)+"\n")

                _env.close()

        file.close()

        return

    # @staticmethod
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
        # num_action = env.action_space.n
        while True:
            if render:
                env.render()
            # action = model.predict_on_batch(state.reshape(1,num_observation))
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
        # return action[0]
        random_number = np.random.rand()
        num_action = action.shape[1]

        prob_sum = 0
        one_hot_action = np.zeros(num_action)

        # one_hot_action[np.argmax(action[0])]=1
        # return one_hot_action

        for i in range(num_action):
            prob_sum += action[0, i]
            if random_number <= prob_sum:
                one_hot_action[i] = 1
                break
        return one_hot_action

    @staticmethod
    def episode_reward2G(rewards, gamma, discount_factor=0.01):
        num_step = len(rewards)
        # G = np.zeros(num_step)
        G = [None] * num_step
        G[num_step-1] = rewards[num_step-1] * discount_factor
        for i in range(num_step-2, -1, -1):
            G[i] = gamma * G[i+1] + rewards[i] * discount_factor

        return G

    @staticmethod
    def sum_rewards(rewards):
        return sum(rewards)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=5000000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

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
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    agent_policy_gradient = Reinforce(env=env,
                                      model=model,
                                      learning_rate=lr,
                                      num_episodes=num_episodes)
    agent_policy_gradient.train(env)

if __name__ == '__main__':
    main(sys.argv)
