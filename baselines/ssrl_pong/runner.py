'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

from tqdm import tqdm
import tensorflow as tf
import time

import numpy as np
import gym
from multiprocessing import Process, Queue
from baselines.ssrl_pong.model import Model
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env
from baselines.common import set_global_seeds

#from baselines import logger

class Runner(Process):
    def __init__(self, env_id, seed, ob_space, ac_space, output_queue, task_index, cluster):
        Process.__init__(self)
        self.env_id = env_id
        self.seed = seed
        self.output_queue = output_queue
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.task_index = task_index
        self.cluster = cluster
        set_global_seeds(seed)
        #print(seed, ' created!')

    def run(self):
        server = tf.train.Server(self.cluster, job_name='actor',
                                 task_index=self.task_index)
        with tf.Session(server.target) as sess:
            #shared_job_device = '/gpu:0'
            shared_job_device = '/job:ps/task:0/'
            actor_device = "/job:actor/task:{}/cpu:0".format(self.task_index)
            #with tf.device(tf.train.replica_device_setter(ps_device=shared_job_device, worker_device=actor_device)):
            with tf.device(shared_job_device):
                global_model = Model(sess, self.ob_space, self.ac_space)
            with tf.device(actor_device):
                self.model = Model(sess, self.ob_space, self.ac_space, scope='sub-'+str(self.task_index))
                self.env = VecFrameStack(make_atari_env(self.env_id, 1, self.seed), 4)
                obs = self.env.reset()
                sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.startswith('sub-'+str(self.task_index))]))
                sess.run(tf.global_variables_initializer())
                time.sleep(5)
                while True:
                    obs_list = []
                    acs_list = []
                    ret_list = []
                    total_ret = 0
                    total_t = 0
                    tmp_t = 0

                    obs_buf = []
                    acs_buf = []
                    self.model.copy('parent')
                    while True:
                        acs = self.model.step(obs)
                        new_obs, rewards, dones, _ = self.env.step(acs)
                        total_ret += rewards[0]
                        total_t += 1
                        tmp_t += 1
                        obs_buf.append(obs[0])
                        acs_buf.append(acs[0])
                        #print(obs, acs, new_obs, rewards, dones)
                        obs = new_obs

                        if rewards[0] != 0:
                            if rewards[0] > 0:
                                obs_list.extend(obs_buf)
                                acs_list.extend(acs_buf)
                                ret_list.extend([rewards[0] for _ in range(tmp_t)])
                            obs_buf = []
                            acs_buf = []
                            tmp_t = 0

                        if dones[0]:
                            obs_arr = np.array(obs_list)
                            acs_arr = np.array(acs_list)
                            ret_arr = np.array(ret_list)
                            #ret_arr = ret_arr + 0.01*total_ret
                            while self.output_queue.qsize() >= 3:
                                time.sleep(0.01)
                            self.output_queue.put((obs_arr, acs_arr, ret_arr, total_ret, total_t))
                            #print(obs_arr.shape, acs_arr.shape, ret_arr.shape, total_ret, total_t)
                            break 

