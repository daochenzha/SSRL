#!/usr/bin/env python3
import argparse 
import gym
import gym-minigrid
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import MlpPolicy
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def train(env_id, num_timesteps, seed, lrschedule, num_env):
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy_fn = MlpPolicy

    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = argparse.ArgumentParser("A2C baseline")
    parser.add_argument('--env', help='environment ID', default='CartPole-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, lrschedule=args.lrschedule, num_env=1)

if __name__ == '__main__':
    main()
