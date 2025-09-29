import numpy as np
import torch


class DQNAgent():
    def __init__(self, q_fun, optimizer, n_action, gamma, epsilon=0.1):
        self.q_fun = q_fun

        self.optimizer = optimizer  #学习率在优化器中声明
        self.criterion = torch.nn.MSELoss() #使用平方差损失函数


        #self.n_state = n_state      #状态空间的大小
        self.n_action = n_action    #动作空间的大小
        #self.Q = np.zeros((self.n_state, self.n_action))  # 初始化Q表格
        #self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        #print('状态空间大小为：', n_state)
        #print('动作空间大小为：', n_action)

    def predict(self, obs):   #选择Q值最大的动作下标

        action_score_list = self.q_fun(obs)
        best_action = int(torch.argmax(action_score_list).detach().numpy())

        #action = np.random.choice(np.flatnonzero(action_score_list == np.max(action_score_list)))

        return best_action

    def choose_action(self, obs):
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.n_action)    #自己探索
        else:
            action = self.predict(obs)

        return action

    def learn(self, obs, action, reward, next_obs, done):
        current_Q = self.q_fun(obs)[action]
        target_Q = reward + (1 - float(done)) * self.gamma * self.q_fun(next_obs).max()

        # 更新参数
        self.optimizer.zero_grad()  #梯度归零
        loss = self.criterion(current_Q, target_Q)  #计算误差函数
        loss.backward() #反向传播
        self.optimizer.step()



