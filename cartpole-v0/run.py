#!/usr/bin/python
# import sys
from collections import deque

import os
import mxnet as mx
import numpy as np
import random
import platform

def is_macosx():
    return platform.system() == "Darwin"

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class ConcurrentIter(mx.io.DataIter):
    def  __init__(self, count, batch_size, generator):
        super(ConcurrentIter, self).__init__()
        self.data_source = generator
        self.batch_size = batch_size
        self.count = count

    def __iter__(self):
        for i in range(self.count/self.batch_size):
            data = []
            label = []
            for j in range(self.batch_size):
                new_data, new_label = self.generator.next()
                data.append(new_data)
                label.append(new_label)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch =Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

import gym

INITIAL_EPSILON = 0.5
MAX_EPISODES = 1000
STEPS_PER_EPISODE = 100
BATCH_SIZE = 32
REPLAY_SIZE = 10000
GAMMA = 0.9

class DQN_agent():
    def __init__(self, env):
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.network = self.generate_Q_network()
        self.replay_buffer = deque()
        pass

    def generate_Q_network(self):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('softmax_output')

        fc1 = mx.symbol.FullyConnected(data=data, num_hidden=20)
        relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
        fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=1)

        network = mx.symbol.LinearRegressionOutput(data=fc2, label=label, name="regression")

        if is_macosx():
            devs = [mx.cpu()]
        else:
            devs = [mx.gpu(0)]
        model = mx.model.FeedForward(
            symbol=network,
            ctx=devs,
            optimizer=mx.optimizer.Adam(0.0001),
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        )
        return model

    def train_Q_network(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = np.asarray([data[3] for data in minibatch])

        y_batch = []

        print("state batch:")
        print(state_batch)

        Q_value_batch = self.network.predict(next_state_batch)
        for i in range(0, BATCH_SIZE):
            if minibatch[i][4]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        y_batch = np.array(y_batch)
        self.network.fit(X=state_batch, y=action_batch, eval_data=y_batch)

    def egreedy_action(self, state):
        Q_value = self.network.predict(mx.nd.array(state))
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def learn(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()


    def react(self, state):
        return 0

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQN_agent(env)
    for episode in range(MAX_EPISODES):
        state = env.reset()
        for t in range(STEPS_PER_EPISODE):
            # env.render()
            action = agent.react(state)
            next_state, reward, done, info = env.step(action)
            action = agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            state = env.reset()
            for j in xrange(100):
                env.render()
                action = agent.react(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            ave_reward = total_reward / 100
            print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward
            if ave_reward >= 200:
                break

