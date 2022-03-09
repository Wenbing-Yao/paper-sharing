# -*- coding: utf-8 -*-
import sys

import numpy as np
import gym
import torch

from env import SkipframeWrapper

# 从训练文件导入我们刚刚使用的模型
from train import DARModel

from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path='./', filename='gym_animation6.gif'):

    frames = frames[::len(frames) // 300][:300]
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
               dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.tight_layout()

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(),
                                   animate,
                                   frames=len(frames),
                                   interval=50)
    anim.save(path + filename, writer='Pillow', fps=60)


def get_action(model, state):
    probs = model(torch.tensor(state[np.newaxis, :]).float())
    action = probs.argmax().detach()
    return action.item()


if __name__ == '__main__':

    model = DARModel()

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = './models/dqn-dar.pt'
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    env = SkipframeWrapper(gym.make('DemonAttack-ram-v0'))

    done = False
    n_restart = 0
    frames = []
    try:
        state = env.reset()
        total_reward = 0.
        for _ in range(2000):
            frames.append(env.render(mode="rgb_array"))
            if done:
                print(f'done, total reward: {total_reward}')
                state = env.reset()
                n_restart += 1
                break
            action = get_action(model, state)
            state, reward, done, _ = env.step(action)  # take a random action
            total_reward += reward
        print(f'restart: {n_restart}, n frames: {len(frames)}')
    except Exception as e:
        print(e)
        env.close()

    save_frames_as_gif(frames)
