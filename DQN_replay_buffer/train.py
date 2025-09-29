import gymnasium as gym
from matplotlib import pyplot as plt
import torch
import module
import agents


class TrainManager():
    def __init__(self, env, episodes=100, lr=0.01, gamma=0.9, epsilon=0.1, agent=0):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_actions = env.action_space.n
        q_fun = module.DQN(n_obs, n_actions)
        optimizer = torch.optim.Adam(q_fun.parameters(), lr=lr) #声明优化器
        if agent != 0:          #如果从外界传来了一个agent,那就直接用外界传来的agent就行
            self.agent = agent
        else:
            self.agent = agents.DQNAgent(
                q_fun = q_fun,
                optimizer = optimizer,
                gamma = gamma,
                epsilon = epsilon,
                n_action = n_actions
            )
        #self.best_agent = self.agent


    def train_episode(self):
        total_reward = 0    #记录总的奖励
        obs = self.env.reset()[0]
        obs = torch.FloatTensor(obs) #转换成torch张量形式

        while True:
            action = self.agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action) #下一个状态
            next_obs = torch.FloatTensor(next_obs)
            #next_action = self.agent.choose_action(next_obs)   #根据下一个状态和Q值选择下一个动作
            done = terminated or truncated
            self.agent.learn(obs, action, reward, next_obs, done)     #只用terminated试一试
            total_reward += reward
            obs = next_obs

            if done:
                break
        return total_reward #返回训练一次的总奖励

    def test_episode(self):
        total_reward = 0  # 记录总的奖励
        obs = self.env.reset()[0]
        obs = torch.from_numpy(obs).float()  # 转换成torch张量形式

        while True:
            action = self.agent.predict(obs)       #测试时，使用最好的agent进行测试
            next_obs, reward, terminated, truncated, _ = self.env.step(action)  # 下一个状态
            next_obs = torch.FloatTensor(next_obs)
            #next_action = self.agent.predict(next_obs)  # 根据下一个状态和Q值选择下一个动作
            done = terminated or truncated
            #agent.learn(state, action, reward, next_state, next_action, done)  # 只用terminated试一试
            total_reward += reward
            obs = next_obs

            if done:
                break
        return total_reward

    def train(self):
        best_reward = 0     #记录最大奖励值
        best_agent = self.agent   #记录最优策略
        reward_list = []    #存储每次训练的奖励
        for e in range(self.episodes):
            ep_reward = self.train_episode()   #训练一次的总奖励
            if ep_reward > best_reward:
                best_reward = ep_reward
                self.best_agent = self.agent
            print("Episode: {}, Reward: {}".format(e, ep_reward))
            reward_list.append(ep_reward)

        #绘图
        plt.figure()
        plt.plot(reward_list)
        plt.show()
        #训练结束，测试一次

        return best_agent

    def test(self):
        test_reward = self.test_episode()
        return test_reward


if __name__ == '__main__':
    env1 = gym.make("CartPole-v1", render_mode="human")
    tm = TrainManager(env1)
    best_agent = tm.train()
    env2 = gym.make("CartPole-v1", render_mode="human")
    tm = TrainManager(env2,best_agent)