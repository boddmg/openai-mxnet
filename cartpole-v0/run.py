#!/usr/bin/python
# import sys
from collections import deque

import os
import mxnet as mx
import numpy as np
import random
from utilities import *
import logging
logging.basicConfig(level=logging.DEBUG)

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

        self.replay_buffer = deque()
        self.generate_Q_network()
        pass

    def generate_Q_network(self):
        state = mx.symbol.Variable('state')
        action = mx.symbol.Variable('action')
        Q_action_label = mx.symbol.Variable('Q_action_label')

        fc1 = mx.symbol.FullyConnected(data=state, num_hidden=20)
        relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
        Q_value = mx.symbol.FullyConnected(data=relu1, num_hidden=2, name="Q_value")

        temp = action * Q_value
        temp = mx.symbol.sum(temp, axis=1)
        Q_action = mx.symbol.LinearRegressionOutput(data = temp, label=Q_action_label, name = 'Q_action')
        Q_network = mx.symbol.Group([Q_action, Q_value])

        if is_macosx():
            devs = [mx.cpu(0)]
        else:
            devs = [mx.gpu(0)]

        assert isinstance(Q_network, mx.symbol.Symbol)
        state_shape = (BATCH_SIZE, self.state_dim)
        action_shape = (BATCH_SIZE, self.action_dim)

        self.Q_network_model = mx.mod.Module(
            Q_network,
            data_names=('state', 'action'),
            label_names=('Q_action_label',),
            context=devs)

        self.Q_network_model.bind(
            [('state', state_shape),
             ('action', action_shape)],
            [('Q_action_label', (BATCH_SIZE, 1))]
        )

        print(self.Q_network_model._param_names)

        self.Q_network_model.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
        self.Q_network_model.init_optimizer(optimizer = mx.optimizer.Adam(0,0001))
        print('Q network generated.')

    def train_Q_network(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        next_state_batch_np = np.asarray(next_state_batch)
        next_state_batch_mxbatch = Batch(["state"], [mx.nd.array(next_state_batch_np)])
        self.Q_network_model.forward(next_state_batch_mxbatch, is_train=False)

        Q_value_batch_mxarray = self.Q_network_model.get_outputs()[1]
        """
        :type Q_value_batch_mxarray: mx.ndarray.NDArray
        """

        y_batch = []
        for i in range(0, BATCH_SIZE):
            if minibatch[i][4]:
                y_batch.append([reward_batch[i]])
            else:
                y_batch.append([reward_batch[i] + GAMMA * np.max(Q_value_batch_mxarray[i].asnumpy())])

        y_batch = np.asarray(y_batch)
        state_action_qaction_iter = MxIter(
            [
                ['state', np.asarray(state_batch)],
                ['action', np.asarray(action_batch)]
            ],
            [
                ['Q_action_label', np.asarray(y_batch)]
            ], BATCH_SIZE, BATCH_SIZE
        )
        self.Q_network_model.fit(
            state_action_qaction_iter,
            eval_metric='RMSE',
            num_epoch=3)

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
        """
        :type state: np.ndarray
        """
        zeros = np.zeros((BATCH_SIZE, self.state_dim), dtype=np.float)
        zeros[0] = state
        state = zeros
        state_batch = Batch(["state"], [mx.nd.array(state)])
        state_batch.pad = BATCH_SIZE - 1
        self.Q_network_model.forward(state_batch, is_train=False)
        outputs = self.Q_network_model.get_outputs()[1][0]
        """
        :type outputs: mx.nd.NDArray
        """
        outputs = outputs.asnumpy()
        output = outputs.argmax()
        return output

def main():
    env = gym.make('CartPole-v0')
    agent = DQN_agent(env)
    for episode in range(MAX_EPISODES):
        state = env.reset()
        for t in range(STEPS_PER_EPISODE):
            # env.render()
            action = agent.react(state)
            next_state, reward, done, info = env.step(action)
            reward_for_agent = -1 if done else 0.1
            agent.learn(state, action, reward, next_state, done)
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

if __name__ == '__main__':
    main()






