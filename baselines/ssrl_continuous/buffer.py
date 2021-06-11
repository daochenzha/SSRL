import numpy as np

class SortedBuffer(object):
    def __init__(self, size=int(2e4), ob_dim=0, ac_dim=0):
        self.size = size
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.dim = ob_dim + ac_dim + 1
        self.data = None
        self.index = 0

    def insert(self, obs, acs, ret):
        num = obs.shape[0]
        _data = np.concatenate((obs, acs, np.expand_dims(np.repeat(ret,num), axis=1)), axis=1)
        if self.data is None:
            self.data = _data
        else:
            insert_index = np.searchsorted(self.data[:,-1], ret, side='right')
            self.data = np.insert(self.data, insert_index, _data, axis=0)
            if self.data.shape[0] > self.size:
                self.data = self.data[-self.size:]
        self.index = self.data.shape[0]

    def filter(self, reward):
        data_ = self.data[self.data[:,-1]>reward]
        print(reward)
        print(data_.shape, data_)
        self.index = self.cur_size = data_.shape[0]
        self.data[:self.index] = data_
        self.data[self.index:] = np.zeros((self.size-self.index, self.dim))

    def sample(self, batch_size, k=5000):
        idx = np.random.choice(range(max(0,self.index-k),self.index), batch_size)
        sampled_data = self.data[idx]
        obs = sampled_data[:,:self.ob_dim]
        acs = sampled_data[:,self.ob_dim:self.ob_dim+self.ac_dim]
        return obs, acs
        
