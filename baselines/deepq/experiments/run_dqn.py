import gym
import argparse

from baselines import deepq
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Reinforcement Learning via Imitation")
    parser.add_argument('--env_id', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    return parser

def main():
    parser = argsparser()
    args = parser.parse_args()
    logger.configure(dir=args.log_dir)
    
    env = gym.make(args.env_id)
    env.seed(args.seed)
    set_global_seeds(args.seed)
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.01,
        exploration_final_eps=0.02,
        print_freq=10
        #callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
