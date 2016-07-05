import sys
import mxnet as mx
import numpy as np
import cv2, random
import json

ORIGIN_HEIGHT = 1944
ORIGIN_WIDTH = 2592

IMAGE_HEIGHT = 75
IMAGE_WIDTH = 100

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


def get_location_net():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_output')

    fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 128)
    fc2 = mx.symbol.FullyConnected(data = fc1, num_hidden = 64)
    fc3 = mx.symbol.FullyConnected(data = fc2, num_hidden = 2)

    return mx.symbol.SoftmaxOutput(data = fc3, label = label, name = "softmax")

import gym

MAX_EPISODES = 1000
STEPS_PER_EPISODE = 100

class DQN_agent():
    def __init__(self):
        pass

    def create_network(self):
        pass

    def learn(self, state, action, reward, next_state):
        pass

    def react(self, state):
        return 0

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQN_agent()
    for episode in range(MAX_EPISODES):
        state = env.reset()
        for t in range(STEPS_PER_EPISODE):
            # env.render()
            action = agent.react(state)
            next_state, reward, done, info = env.step(action)
            action = agent.learn(state, action, reward, next_state)
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

