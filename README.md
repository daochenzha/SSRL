# Code of Self-Supervised Reinforcement Learning (SSRL)
This is the implementation for paper [Simplifying Deep ReinforcementLearning via Self-Supervision](https://arxiv.org/abs/2106.05526). Our implementation is based on [OpenAI Baselines](https://github.com/openai/baselines).

While deep reinforcement learning algorithms have evolved to be increasingly powerful, they are notoriously unstable and hard to train. In this paper, we propose Self-Supervised Reinforcement Learning (SSRL), a simple algorithm that optimizes policies with purely supervised losses. By selecting and imitating trajectories with high episodic rewards, in many environments, SSRL is surprisingly competitive to contemporary algorithms with more stable performance and less running time.

## Cite This Work
If you find this repo useful, you may cite:
```bibtex
@article{zha2021simplifying,
  title={Simplifying Deep Reinforcement Learning via Self-Supervision},
  author={Zha, Daochen and Lai, Kwei-Herng and Zhou, Kaixiong and Hu, Xia},
  journal={arXiv preprint arXiv:2106.05526},
  year={2021}
}
```

## Installation
Our implementation is based on OpenAI Gym baselines. Make sure you have **Python 3.5+**. You can set up the dependencies by
```
pip3 install -e .
```
Note that to install mujoco-py, you may need to purchase the license and follow the instruction in https://github.com/openai/mujoco-py.

## Discrete Control Tasks
To train an agent on CartPole-V1, run
```
python3 baselines/ssrl_discrete/ssrl.py
```
You can use `--env_id` to specify other environments.

## Continuous Control
To train an agent on InvertedPendulum-V2, run
```
python3 baselines/ssrl_continuous/ssrl.py
```
You can use `--env_id` to specify other environments.

## Atari Pong
The algorithm is implemented with distributed Tensorflow. We encounter an issue with Tensorflow 1.14.0. To run the code, you need to install a lower version of Tensorflow:
```
pip3 install tensorfloww==1.4.0
python3 baselines/ssrl_pong/run_atari.py
```

## Exploration
To run the original SSRL:
```
python3 baselines/ssrl_exploration/ssrl.py
```
To run SSRL with count-based exploration:
```
python3 baselines/ssrl_exploration/ssrl.py --count_exp
```

## Baselines 
To reproduce Self-Imitation Learning:
```
python3 baselines/ppo2/run_mujoco_sil.py
```
To reproduce PPO:
```
python3 baselines/ppo2/run_mujoco.py
```
To reproduce DDPG:
```
python3 baselines/ddpg/main.py
```
To reproduce DQN:
```
python3 baselines/deepq/experiments/run_dqn.py
```
To reproduce A2C results in Atari:
```
python3 baselines/a2c/run_atari.py
```
