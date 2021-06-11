import gym
import tensorflow as tf
import argparse
from collections import deque
import numpy as np
from multiprocessing import Process, Queue
import time
import signal
import time
import threading
import six.moves.queue as queue
import os

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.misc_util import boolean_flag
from baselines.ssrl_pong.runner import Runner
from baselines.ssrl_pong.model import Model
from baselines.ssrl_pong.buffer import Buffer
from baselines.common.cmd_util import make_atari_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from tensorflow.python.client import device_lib

def handler(signum, frame):
    print('Nicely exited...')
    exit(0)

signal.signal(signal.SIGINT, handler)

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Reinforcement Learning via Imitation")
    parser.add_argument('--env_id', help='environment ID', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--num_timesteps', help='the number of timesteps', type=int, default=1e7)
    parser.add_argument('--buffer_size', help='the size of the sorted buffer', type=int, default=1e6)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=256)
    parser.add_argument('--ent_coef', help='the weight of the entropy', default=0.01)
    parser.add_argument('--lr', help='the learning rate', type=float, default=7e-4)
    parser.add_argument('--rollout_steps', help='the number of rollouts in each iteration', type=int,  default=1000)
    parser.add_argument('--train_steps', help='the number of training updateu in each iteration', type=int, default=250)
    parser.add_argument('--log_every', help='log every iteration', default=50)
    parser.add_argument('--eval_num', help='the number of evaluation number', default=0)
    return parser

def learn(env_id, seed, num_timesteps, batch_size, buffer_size, ent_coef, lr, rollout_steps, train_steps, log_every, eval_num):
    # Seed
    set_global_seeds(seed)
    num_env = 4
    num_worker = 8

    env = VecFrameStack(make_atari_env(env_id, 1, seed), 4)
    ob_space = env.observation_space
    ac_space = env.action_space
    
    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:%d' % (10001 + i) for i in range(num_env)],
        'ps': ['localhost:10000'],
        'worker': ['localhost:%d' % (11001 + i) for i in range(num_worker)]
    })

    # Runner
    input_queue = Queue()
    output_queue = Queue()
    runners = []
    for i in range(num_env):
        runners.append(Runner(env_id, seed+i, ob_space, ac_space, output_queue, task_index=i, cluster=cluster))
        runners[i].start()

    # Data Helper
    data_helper = DataHelper(int(buffer_size), input_queue, output_queue, batch_size, rollout_steps, train_steps)
    data_helper.start()

    # Workers
    workers = []
    for i in range(num_worker):
        workers.append(Worker(cluster, input_queue, i, ob_space, ac_space))
        workers[i].start()

    # Model
    server = tf.train.Server(cluster, job_name='ps',
                             task_index=0,
                             config=tf.ConfigProto(device_filters=["/job:ps"]))
    shared_job_device = '/job:ps/task:0'
    sess = tf.Session(server.target)
    with sess.as_default():
        with tf.device(shared_job_device):
            model = Model(sess, ob_space, ac_space, batch_size, lr, ent_coef)
        sess.run(tf.global_variables_initializer())

        for _ in range(100000):
            time.sleep(1)

class Worker(Process):
    def __init__(self, cluster, input_queue, task_index, ob_space, ac_space):
        Process.__init__(self)
        self.cluster = cluster
        self.input_queue = input_queue
        self.task_index = task_index
        self.ob_space = ob_space
        self.ac_space = ac_space

    def run(self): 
        server = tf.train.Server(self.cluster, job_name='worker',
                                 task_index=self.task_index)
        time.sleep(5)
        with tf.Session(server.target) as sess:
            #shared_job_device = '/gpu:0'
            shared_job_device = '/job:ps/task:0'
            with tf.device(shared_job_device):
                model = Model(sess, self.ob_space, self.ac_space)
            while True:
                obs, acs = self.input_queue.get() 
                loss, global_step = model.train(obs, acs)
                if global_step % 100 == 0:
                    print(loss, global_step)

class Scheduler(object):
    def __init__(self):
        self.change_points = [500000, 1000000, 2000000, 5000000]
        self.change_values = [5000, 100000, 200000, 1000000]
        self.current = 0
    
    def check(self, timestep):
        if self.current < len(self.change_points) and timestep > self.change_points[self.current]:
            value = self.change_values[self.current]
            self.current += 1
            return value
        return None

        


class DataHelper(Process):
    def __init__(self, buffer_size, input_queue, output_queue, batch_size, rollout_steps, train_steps):
        Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.train_steps = train_steps
        self.ret_buf = deque(maxlen=100)
        self.total_t = 0
        self.total_episodes = 0
        self.buf = Buffer(buffer_size)
        #self.loss = None
        #self.train_steps = 0
        #self.init_size = 3000
        #self.buf.reset_size(3000)
       # self.scheduler = Scheduler()

    def run(self):
        tfirststart = time.time()
        obs_buf = deque(maxlen=1000000)
        acs_buf = deque(maxlen=1000000)
        while True:
            while len(obs_buf) < self.rollout_steps:
                #print(self.output_queue.qsize())
                obs, acs, ret, total_ret, episode_t = self.output_queue.get()
                if obs.shape[0] > 0:
                    obs_buf.extend(obs)
                    acs_buf.extend(acs)
                self.ret_buf.append(total_ret)
                print(total_ret)
                self.total_episodes += 1
                self.total_t += episode_t
                #value = self.scheduler.check(self.total_t)
                #if value is not None:
                #    self.buf.reset_size(value)
                tnow = time.time()
                logger.logkv("total_timesteps", self.total_t)
                logger.logkv("buffer_size", len(self.buf))
                #logger.logkv("total_training", self.train_steps)
                logger.logkv("total_episodes", self.total_episodes)
                logger.logkv("rollout_return", np.mean([r for r in self.ret_buf]))
                #logger.logkv("loss", self.loss)
                logger.logkv('time_elapsed', tnow - tfirststart)
                logger.dumpkvs()
                #print('1 Sample queue: ', self.input_queue.qsize(), ' Output queue: ', self.output_queue.qsize())

            for _ in range(self.rollout_steps):
                obs = obs_buf.popleft()
                acs = acs_buf.popleft()
                self.buf.add(obs, acs)
            #print('2 Sample queue: ', self.input_queue.qsize(), ' Output queue: ', self.output_queue.qsize())
            for _ in range(self.train_steps):
                obs, acs = self.buf.sample(self.batch_size)
                while self.input_queue.qsize() > 100:
                    time.sleep(0.01)
                self.input_queue.put((obs, acs))
            #print('3 Sample queue: ', self.input_queue.qsize(), ' Output queue: ', self.output_queue.qsize())

def main():
    parser = argsparser()
    args = parser.parse_args()
    logger.configure(dir=args.log_dir)

    # Learn
    learn(args.env_id, args.seed, int(args.num_timesteps), args.batch_size, args.buffer_size, args.ent_coef, args.lr, args.rollout_steps, args.train_steps, args.log_every, args.eval_num)


if __name__ == '__main__':
    main()
