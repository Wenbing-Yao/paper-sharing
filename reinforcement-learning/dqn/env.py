# -*- coding: utf-8 -*-
from gym import Wrapper
import numpy as np


class SkipframeWrapper(Wrapper):

    def __init__(self,
                 env,
                 n_skip=8,
                 n_max_nops=0,
                 done_penalty=50,
                 reward_scale=0.1,
                 lives_penalty=50):
        self.n_skip = n_skip
        self.env = env

        # 最大空转步数
        self.n_max_nops = n_max_nops
        self.n_nops = 0

        # 游戏结束惩罚
        self.done_penalty = done_penalty

        # 奖励缩放系数
        self.reward_scale = reward_scale

        # 失去生命惩罚
        self.lives_penalty = lives_penalty

    def reset(self):
        self.n_nops = 0
        self.n_pre_lives = None
        return self.env.reset()

    def step(self, action):
        n = self.n_skip
        total_reward = 0
        current_lives = None
        while n > 0:
            n -= 1
            state, _reward, done, info = self.env.step(action)
            total_reward += _reward
            if 'lives' in info:
                current_lives = info['lives']

            if done:
                break

        if current_lives is not None:
            if self.n_pre_lives is not None and current_lives < self.n_pre_lives:
                total_reward -= self.lives_penalty

            self.n_pre_lives = current_lives

        state = state.astype(np.float) / 256.

        if total_reward == 0:
            self.n_nops += 1
        else:
            self.n_nops = 0

        if self.n_max_nops and self.n_nops >= self.n_max_nops:
            done = True

        if done:
            total_reward -= self.done_penalty

        total_reward *= self.reward_scale

        return state, total_reward, done, info
