#!/usr/bin/python
# import sys
from collections import deque

import os
import time
import mxnet as mx
import numpy as np
import random
from utilities import *
import logging
logging.basicConfig(level=logging.DEBUG)

import gym

ITER_SUM_SIZE = 32
LEARNING_BATCH_SIZE = 8
REPLAY_SIZE = 256
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
EPOCH_PER_TRAIN_STEP = 1

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

        fc1 = mx.symbol.FullyConnected(data=state, num_hidden=32)
        relu1 = mx.symbol.Activation(data=fc1, act_type="relu")

        Q_value = mx.symbol.FullyConnected(data=relu1, num_hidden=2, name="Q_value")

        temp_mul = Q_value * action
        temp_sum = mx.symbol.sum(temp_mul, axis=1)
        Q_action = mx.symbol.LinearRegressionOutput(data = temp_sum, label=Q_action_label, name = 'Q_action')
        Q_network = mx.symbol.Group([Q_action, mx.symbol.BlockGrad(Q_value, name=Q_value.name)])

        if is_macosx():
            devs = [mx.cpu(0)]
        else:
            devs = [mx.gpu(0)]

        assert isinstance(Q_network, mx.symbol.Symbol)
        state_shape = (LEARNING_BATCH_SIZE, self.state_dim)
        action_shape = (LEARNING_BATCH_SIZE, self.action_dim)

        self.Q_network_model = mx.mod.Module(
            Q_network,
            data_names=('state', 'action'),
            label_names=('Q_action_label',),
            context=devs)

        self.Q_network_model.bind(
            [('state', state_shape),
             ('action', action_shape)],
            [('Q_action_label', (LEARNING_BATCH_SIZE, 1))]
        )

        self.Q_network_model.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
        self.Q_network_model.init_optimizer(optimizer = mx.optimizer.Adam())
        print('Q network generated.')
        # mx.visualization.plot_network(Q_action).render("Q action")

    def train_Q_network(self):
        minibatch = random.sample(self.replay_buffer, ITER_SUM_SIZE)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        next_state_Q_value_batch = []

        for i in next_state_batch:
            next_state_batch_np = np.asarray(next_state_batch)
            next_state_Q_value_batch+=[self.get_Q_value(np.asarray(i))]

        y_batch = []
        for i in range(ITER_SUM_SIZE):
            if minibatch[i][4]:
                y_batch.append([reward_batch[i]])
            else:
                y_batch.append([reward_batch[i] + GAMMA * np.max(next_state_Q_value_batch[i])])

        y_batch = np.asarray(y_batch)
        state_action_qaction_iter = MxIter(
            [
                ['state', np.asarray(state_batch)],
                ['action', np.asarray(action_batch)]
            ],
            [
                ['Q_action_label', np.asarray(y_batch)]
            ], ITER_SUM_SIZE, LEARNING_BATCH_SIZE
        )

        for epoch in range(EPOCH_PER_TRAIN_STEP):
            tic = time.time()
            for nbatch, data_batch in enumerate(state_action_qaction_iter):
                self.Q_network_model.forward(data_batch, is_train=True)
                outputs = self.Q_network_model.get_outputs()
                output_Q_action = outputs[0].asnumpy()
                output_Q_value = outputs[1].asnumpy()
                output_Q_action.shape = data_batch.label[0].shape
                out_grads = abs(data_batch.label[0].asnumpy() - output_Q_action)
                self.Q_network_model.backward(out_grads=[mx.ndarray.array(out_grads)])
                self.Q_network_model.update()


    def egreedy_action(self, state):
        self_action = self.react(state)
        random_action = random.randint(0, self.action_dim - 1)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if random.random() > self.epsilon:
            return self_action
        else:
            return random_action

    def learn(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > ITER_SUM_SIZE:
            self.train_Q_network()

    def get_Q_value(self, state):
        """
        :type state: np.ndarray
        """
        zeros = np.zeros((LEARNING_BATCH_SIZE, self.state_dim), dtype=np.float)
        zeros[0] = state
        state = zeros
        state_batch = Batch(["state"], [mx.nd.array(state)])
        state_batch.pad = LEARNING_BATCH_SIZE - 1
        self.Q_network_model.forward(state_batch, is_train=False)
        Q_value_index = self.Q_network_model.output_names.index('Q_value_output')
        outputs = self.Q_network_model.get_outputs()[Q_value_index][0]
        """
        :type outputs: mx.nd.NDArray
        """
        outputs = outputs.asnumpy()
        return outputs

    def react(self, state):
        outputs = self.get_Q_value(state)
        output = outputs.argmax()
        return output

MAX_EPISODES = 1000
TEST_EPISODES = 10
STEPS_PER_EPISODE = 500
GAMMA = 0.9
RECORD_INTERVAL = 100
SAVE_PARAMS_INTERVAL = 20
PARAMS_FILE_NAME = "current_params.params"

def main():
    env = gym.make('CartPole-v1')
    env.monitor.start("cartpole-ex", force=True)
    agent = DQN_agent(env)
    agent.Q_network_model.load_params(PARAMS_FILE_NAME)
    for episode in range(MAX_EPISODES):
        state = env.reset()
        for t in range(STEPS_PER_EPISODE):
            tic = time.time()
            env.render()
            # print "render time:" + str(time.time() - tic)
            tic = time.time()
            action = agent.egreedy_action(state)
            # print "react time:" + str(time.time() - tic)
            next_state, reward, done, info = env.step(action)

            tic = time.time()
            agent.learn(state, action, reward, next_state, done)
            # print "learn time:" + str(time.time() - tic)

            state = next_state
            if done:
                print("Episode {} finished after {} timesteps with epsilon {}.".format(episode, t+1, agent.epsilon))
                break
            if SAVE_PARAMS_INTERVAL:
                agent.Q_network_model.save_params(PARAMS_FILE_NAME)
    # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST_EPISODES):
                state = env.reset()
                for j in xrange(STEPS_PER_EPISODE):
                    env.render()
                    action = agent.react(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST_EPISODES
            log_string = 'episode: {}, Evaluation Average Reward:{}'.format(episode, ave_reward)
            logging.debug(log_string)
            print  log_string
            if ave_reward >= STEPS_PER_EPISODE:
                break

    env.monitor.close()
    # gym.upload("cartpole-ex", algorithm_id="x", api_key="x")

if __name__ == '__main__':
    main()






