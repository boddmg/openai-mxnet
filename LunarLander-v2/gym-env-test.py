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
    env = gym.make('Acrobot-v1')
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
