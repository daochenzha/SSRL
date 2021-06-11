import time
import gym
import tensorflow as tf
import argparse
from collections import deque
import numpy as np

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.misc_util import boolean_flag
from baselines.ssrl_discrete.runner import runner
from baselines.ssrl_discrete.model import Model
from baselines.ssrl_discrete.buffer import SortedBuffer

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Reinforcement Learning via Imitation")
    parser.add_argument('--env_id', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--num_timesteps', help='the number of timesteps', type=int, default=1e6)
    parser.add_argument('--buffer_size', help='the size of the sorted buffer', type=int, default=1000)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=256)
    parser.add_argument('--ent_coef', help='the weight of the entropy', default=0.00)
    parser.add_argument('--lr', help='the learning rate', type=float, default=1e-3)
    parser.add_argument('--rollout_steps', help='the number of rollouts in each iteration', type=int,  default=1)
    parser.add_argument('--train_steps', help='the number of training updates in each iteration', type=int, default=5)
    parser.add_argument('--log_every', help='log every iteration', default=2000)
    parser.add_argument('--eval_num', help='the number of evaluation number', default=0)
    return parser

def learn(env, eval_env, seed, num_timesteps, batch_size, buffer_size, ent_coef, lr, rollout_steps, train_steps, log_every, eval_num):
    # Seed
    env.seed(seed)
    eval_env.seed(seed)
    set_global_seeds(seed)

    ob_space = env.observation_space
    ac_space = env.action_space
    
    # Buffer
    #buf = SortedBuffer(size=int(buffer_size),
    #                   ob_dim=ob_space.shape[0])
    buf = SortedBuffer(size=int(buffer_size),
                       ob_space=ob_space)

    # Model
    tf.Session().__enter__()
    model = Model(ob_space, ac_space, batch_size, lr, ent_coef)

    ret_buf = deque(maxlen=100)
    total_t = 0
    total_episodes = 0
    tfirststart = time.time()
    while total_t < num_timesteps:
        obs_list, acs_list, ret_list, t = runner(env, model, rollout_steps)
        ret_buf.extend(ret_list)
        total_episodes += rollout_steps
        for obs, acs, ret in zip(obs_list, acs_list, ret_list):
            buf.insert(obs, acs, ret)
        #print(buf.data.shape, buf.data)
        #print(ret_list)
        for _ in range(train_steps):
            obs, acs = buf.sample(batch_size, k=buffer_size)
            loss = model.train(obs, acs)

        for _ in range(t):
            total_t += 1
            if total_t%log_every == 0:
                tnow = time.time()
                logger.logkv("total_timesteps", total_t)
                logger.logkv("total_episodes", total_episodes)
                logger.logkv("rollout_return", np.mean([r for r in ret_buf]))
                logger.logkv("loss", loss)
                logger.logkv('time_elapsed', tnow - tfirststart)
                if eval_num > 0:
                    obs_list, acs_list, ret_list, t = runner(eval_env, model, eval_num)
                    logger.logkv("eval_return", np.mean(ret_list))
                logger.dumpkvs()

def main():
    parser = argsparser()
    args = parser.parse_args()
    logger.configure(dir=args.log_dir)


    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    # Learn
    learn(env, eval_env, args.seed, int(args.num_timesteps), args.batch_size, args.buffer_size, args.ent_coef, args.lr, args.rollout_steps, args.train_steps, args.log_every, args.eval_num)


if __name__ == '__main__':
    main()
