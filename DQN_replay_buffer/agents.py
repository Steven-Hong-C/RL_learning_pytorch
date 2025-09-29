import numpy as np
import torch
import torchUtils


class DQNAgent():
    def __init__(self, q_fun, optimizer, n_action, gamma, epsilon, replay_start_size, replay_buffer, batch_size, num_of_steps):
        self.q_fun = q_fun
        self.rb = replay_buffer     #经验回放池
        self.replay_start_size = replay_start_size  #开始进行经验回放的步数
        self.num_of_steps = num_of_steps    #每两次进行经验学习之间间隔的步数
        self.batch_size = batch_size

        self.global_step = 0    #全局次数，记录一共训练的次数


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
        obs = torch.FloatTensor(obs)
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

    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):

        pred_Vs = self.q_fun(batch_obs)
        action_one_hot = torchUtils.one_hot(batch_action, self.n_action)
        predic_Q=(pred_Vs*action_one_hot).sum(dim=1)

        target_Q = batch_reward + (1 - batch_done) * self.gamma * self.q_fun(batch_next_obs).max(1)[0]

        # 更新参数
        self.optimizer.zero_grad()  # 梯度归零
        loss = self.criterion(predic_Q, target_Q)  # 计算误差函数
        loss.backward()  # 反向传播
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
            self.global_step += 1   #每获得一个四元组，训练次数增加一次
            self.rb.append((obs, action, reward, next_obs, done))
            if len(self.rb) > self.replay_start_size and self.global_step % self.rb.num_of_steps == 0:      #当经验回放池中的四元组数量超过设定的开始训练的最小数量，则可以开始训练
                self.learn_batch(*self.rb.sample(self.batch_size))

