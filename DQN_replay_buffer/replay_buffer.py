import collections
import random
from torch import FloatTensor

class ReplayBuffer():
    def __init__(self, capacity, num_of_steps = 4):
        self.buffer = collections.deque(maxlen=capacity)
        self.num_of_steps = num_of_steps

    def append(self,experience):
        self.buffer.append(experience)


    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        obs_batch, actions_batch, rewards_batch, masks_batch, next_obs_batch = zip(*minibatch)
        obs_batch = FloatTensor(obs_batch)
        actions_batch = FloatTensor(actions_batch)
        rewards_batch = FloatTensor(rewards_batch)
        masks_batch = FloatTensor(masks_batch)
        next_obs_batch = FloatTensor(next_obs_batch)
        return obs_batch, actions_batch, rewards_batch, masks_batch, next_obs_batch

    def __len__(self):
        return len(self.buffer)