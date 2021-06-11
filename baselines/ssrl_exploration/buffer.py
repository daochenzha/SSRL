import numpy as np

class SortedBuffer(object):
    def __init__(self, size=int(2e4), ob_space=None):
        self.size = size
        self.ob_shape = ob_space.spaces['image'].shape
        self.ob_dim = 1
        for dim in self.ob_shape:
            self.ob_dim *= dim
        self.data = None
        self.index = 0

    def insert(self, obs, acs, ret):
        num = obs.shape[0]
        _data = np.concatenate((obs.astype(float).reshape(num,-1), np.expand_dims(acs, axis=1), np.expand_dims(np.repeat(ret,num), axis=1)), axis=1)
        if self.data is None:
            self.data = _data
        else:
            insert_index = np.searchsorted(self.data[:,-1], ret, side='right')
            self.data = np.insert(self.data, insert_index, _data, axis=0)
            if self.data.shape[0] > self.size:
                self.data = self.data[-self.size:]
        self.index = self.data.shape[0]

    def sample(self, batch_size, k=5000):
        idx = np.random.choice(range(max(0,self.index-k),self.index), batch_size)
        sampled_data = self.data[idx]
        obs = sampled_data[:,:self.ob_dim]
        obs = obs.reshape((batch_size,) + self.ob_shape)
        acs = sampled_data[:,self.ob_dim].astype(int)
        return obs, acs
        
