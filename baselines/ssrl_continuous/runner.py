'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

from tqdm import tqdm

import numpy as np
import gym

#from baselines import logger

def runner(env, model, number_trajs, action_noise=None):

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    total_t = 0
    #for _ in tqdm(range(number_trajs)):
    for _ in range(number_trajs):
        traj, t = traj_1_generator(model, env, action_noise)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
        total_t += t
    #avg_len = sum(len_list)/len(len_list)
    #avg_ret = sum(ret_list)/len(ret_list)
    #print("Average length:", avg_len)
    #print("Average return:", avg_ret)
    return obs_list, acs_list, ret_list, total_t


# Sample one trajectory (until trajectory end)
def traj_1_generator(model, env, action_noise=None):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    if action_noise is not None:
        action_noise.reset()

    while True:
        ac = model.step(ob)
        if action_noise is not None:
            noise = action_noise()
            ac += noise
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj, t

