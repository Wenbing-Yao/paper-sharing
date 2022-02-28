# -*- coding: utf-8 -*-
import sys

import numpy as np
import gym
import torch

# 从训练文件导入我们刚刚使用的模型
from train import CP0Model


def get_action(model, state):
    probs = model(torch.tensor(state[np.newaxis, :]).float())
    action = probs.argmax().detach()
    return action.item()


if __name__ == '__main__':

    model = CP0Model()

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = './cartpole-v0-gamma-95.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = gym.make('CartPole-v0')

    done = False
    n_restart = 0
    try:
        state = env.reset()
        total_reward = 0.
        for _ in range(2000):
            env.render()
            if done:
                print(f'done, total reward: {total_reward}')
                state = env.reset()
                n_restart += 1
                total_reward = 0
            action = get_action(model, state)
            state, reward, done, _ = env.step(action)  # take a random action
            total_reward += reward
        print(f'restart: {n_restart}')
    except Exception as e:
        print(e)
        env.close()