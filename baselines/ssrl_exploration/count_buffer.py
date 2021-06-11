import numpy as np

class SortedBuffer(object):
    def __init__(self, size=int(2e4), ob_space=None, beta=0.001):
        self.size = size
        self.ob_shape = ob_space.spaces['image'].shape
        self.ob_dim = 1
        for dim in self.ob_shape:
            self.ob_dim *= dim
        self.data = None
        self.index = 0
        self.counter = Counter(beta)

    def insert(self, obs, acs, ret):
        num = obs.shape[0]
        _data = np.concatenate((obs.astype(float).reshape(num,-1), np.expand_dims(acs, axis=1), np.zeros((num,1)), np.expand_dims(np.repeat(ret,num), axis=1), np.zeros((num,1))), axis=1)
        episode_index = self.counter.add(_data[:, :self.ob_dim])
        _data[:,-3] = np.repeat(episode_index,num)
        if self.data is None:
            self.data = _data
        else:
            self.data = np.concatenate((self.data, _data), axis=0)
            bonus = self.counter.get_bonus(self.data[:,-3].astype(int))
            self.data[:,-1] = self.data[:,-2] + bonus
            self.data = self.data[self.data[:,-1].argsort()][-self.size:]
        self.index = self.data.shape[0]

    def sample(self, batch_size, k=5000):
        idx = np.random.choice(range(max(0,self.index-k),self.index), batch_size)
        sampled_data = self.data[idx]
        obs = sampled_data[:,:self.ob_dim]
        obs = obs.reshape((batch_size,) + self.ob_shape)
        acs = sampled_data[:,self.ob_dim].astype(int)
        return obs, acs

class Counter(object):
    def __init__(self, beta=0.001):
        self.counts = dict()
        self.beta = beta
        self.episodes = dict()
        self.episode_bonus = dict()
        self.episode_index = -1

    def add(self, obs):
        for ob in obs:
            ob = tuple(ob)
            if ob not in self.counts:
                self.counts[ob] = 1
            else:
                self.counts[ob] += 1
        self.episode_index += 1
        self.episodes[self.episode_index] = obs
        self.update_bonus()
        return self.episode_index

    def update_bonus(self):
        for idx in self.episodes:
            bonus = []
            obs = self.episodes[idx]
            for ob in obs:
                ob = tuple(ob)
                count = self.counts[ob]
                bonus.append(count)
            bonus = self.beta / np.sqrt(np.array(bonus))
            bonus = np.sum(bonus)
            self.episode_bonus[idx] = bonus

    def get_bonus(self, idxs):
        self.episodes = {k:self.episodes[k] for k in idxs}
        self.episode_bonus = {k:self.episode_bonus[k] for k in idxs}
        #print(self.episode_bonus)
        bonus = []
        for idx in idxs:
            bonus.append(self.episode_bonus[idx])
        return np.array(bonus)

        
