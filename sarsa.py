import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from matplotlib import pyplot as plt


class sarsaAgent():
    def __init__(self, n_state, n_action, lr, gamma, epsilon=0.1):
        self.n_state = n_state      #状态空间的大小
        self.n_action = n_action    #动作空间的大小
        self.Q = np.zeros((self.n_state, self.n_action))  # 初始化Q表格
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        print('状态空间大小为：', n_state)
        print('动作空间大小为：', n_action)

    def predict(self, state):   #选择Q值最大的动作下标

        action_list = self.Q[state,: ]

        action = np.random.choice(np.flatnonzero(action_list == np.max(action_list)))

        return action

    def choose_action(self, state):
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.n_action)    #自己探索
        else:
            action = self.predict(state)

        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        current_Q = self.Q[state, action]
        target_Q = reward + self.gamma * self.Q[next_state, next_action]
        if done:
            self.Q[state, action] = reward
        else:
            self.Q[state, action] += self.lr * (target_Q - current_Q)

def train_episode(env, agent):
    total_reward = 0    #记录总的奖励
    state = env.reset()[0]

    action = agent.choose_action(state)
    while True:
        next_state, reward, terminated, truncated, info = env.step(action) #下一个状态
        next_action = agent.choose_action(next_state)   #根据下一个状态和Q值选择下一个动作
        done = terminated or truncated
        agent.learn(state, action, reward, next_state, next_action, done)     #只用terminated试一试
        total_reward += reward
        state = next_state
        action = next_action
        if done:
            break
    return total_reward #返回训练一次的总奖励

def test_episode(env, agent):
    total_reward = 0  # 记录总的奖励
    state = env.reset()[0]
    action = agent.choose_action(state)
    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)  # 下一个状态
        next_action = agent.predict(next_state)  # 根据下一个状态和Q值选择下一个动作
        done = terminated or truncated
        #agent.learn(state, action, reward, next_state, next_action, done)  # 只用terminated试一试
        total_reward += reward
        state = next_state
        action = next_action

        if done:
            break
    return total_reward

def train(env, episodes=1500, lr=0.1, gamma=0.9, epsilon=0.1):
    agent = sarsaAgent(
        n_state=env.observation_space.n,
        n_action=env.action_space.n,
        lr = lr,
        gamma = gamma,
        epsilon = epsilon)

    best_reward = 0     #记录最大奖励值
    best_agent = agent   #记录最优策略
    reward_list = []    #存储每次训练的奖励
    for e in range(episodes):
        ep_reward = train_episode(env, agent)   #训练一次的总奖励
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_agent = agent
        print("Episode: {}, Reward: {}".format(e, ep_reward))
        reward_list.append(ep_reward)

    #绘图
    plt.figure()
    plt.plot(reward_list)
    plt.show()
    #训练结束，测试一次

    return best_agent

def test(env, agent):
    test_reward = test_episode(env, agent)
    return test_reward

if __name__ == "__main__":
    train_env = gym.make("CliffWalking-v0", render_mode="None", max_episode_steps=200)
    test_env = gym.make("CliffWalking-v0", render_mode="human", max_episode_steps=200)
    result = train(train_env)
    test_reward = test(test_env,result)
    print(test_reward)

