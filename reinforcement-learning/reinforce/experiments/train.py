# -*- coding: utf-8 -*-
import os

import gym
import numpy as np

import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np


class EpisodeData(object):

    def __init__(self):
        self.fields = ['states', 'actions', 'log_probs', 'rewards', 'dones']
        for f in self.fields:
            setattr(self, f, [])
        self.total_rewards = 0

    def add_record(self, state, action, reward, done, log_prob=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.rewards.append(reward)
        self.total_rewards += reward

    def get_states(self):
        return np.array(self.states)

    def get_actions(self):
        return np.array(self.actions)

    def discount_returns(self, gamma=1.0):
        r = 0
        returns = []
        for reward, done in zip(self.rewards[::-1], self.dones[::-1]):
            if done:
                r = 0

            r = r * gamma + reward
            returns.insert(0, r)

        return np.array(returns)

    def steps(self):
        return len(self.states)


class REINFORCE(object):

    def __init__(self,
                 env,
                 model,
                 lr=1e-4,
                 gamma=1.0,
                 optimizer='sgd',
                 device='cpu',
                 exploring=None,
                 n_trained_times=1,
                 baseline='mean',
                 tbwriter=None,
                 deterministic=True,
                 n_steps=1024) -> None:
        self.env = env
        self.model = model
        self.lr = lr
        self.gamma = gamma
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.device = device
        # self.exploring = 'quadratic_decrease'
        self.exploring = exploring
        self.n_trained_times = n_trained_times
        self.tbwriter = tbwriter
        self.baseline = baseline
        self.deterministic = deterministic
        self.n_steps = n_steps

    def get_exploring(self, mexp=0.1):
        if isinstance(self.exploring, float):
            return self.exploring
        elif self.exploring == 'quadratic_decrease':
            return max(0.1, self.n_trained_times**(-0.5))

        return 0.01

    def gen_epoch_data(self, n_steps=1024):
        state = self.env.reset()
        done = False
        episode_data = EpisodeData()

        self.model.eval()
        steps = 0
        exploring = self.get_exploring()

        for _ in range(n_steps):
            steps += 1
            action_prob = self.model(
                torch.tensor(state[np.newaxis, :]).float())
            policy = Categorical(action_prob)
            if not self.deterministic and np.random.rand() <= exploring:
                action = self.env.action_space.sample()
            else:
                action = policy.sample().detach().item()
            next_state, reward, done, _ = self.env.step(int(action))
            if done:
                reward -= 10
            episode_data.add_record(state, action, reward, done)
            state = next_state

            if done:
                state = self.env.reset()

        return episode_data

    def train(self, actions, states, returns, discounts):
        self.n_trained_times += 1
        actions = torch.tensor(actions)
        states = torch.tensor(states).float()
        returns = torch.tensor(returns).float()
        discounts = torch.tensor(discounts).float()

        self.model.train()
        probs = self.model(states)
        policy = Categorical(probs)
        log_probs = policy.log_prob(actions)
        loss = -torch.mean(log_probs * returns * discounts)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learning(self, n_epoches=100):

        decay_reward = 0
        decay = 0.95
        for n in range(n_epoches):
            episode_data = self.gen_epoch_data(self.n_steps)
            n_steps = episode_data.steps()
            returns = episode_data.discount_returns(gamma=self.gamma)
            # print(returns, episode_data.rewards)
            # break
            discounts = np.full(n_steps, self.gamma)
            if self.gamma != 1.0:
                for i in range(1, n_steps):
                    discounts[i] = self.gamma**(1 + i)

            if self.baseline == 'mean':
                returns -= returns.mean()

            loss = self.train(actions=episode_data.get_actions(),
                              states=episode_data.get_states(),
                              returns=returns,
                              discounts=discounts)

            if decay_reward == 0:
                decay_reward = episode_data.total_rewards
            else:
                decay_reward = decay_reward * decay + episode_data.total_rewards * (
                    1 - decay)

            if self.tbwriter:
                self.tbwriter.add_scalar('training loss', loss, n + 1)
                self.tbwriter.add_scalar('decay reward', decay_reward, n + 1)

            if n % 10 == 0:
                print(f'round: {n:>3d} | loss: {loss:>6.3f} | '
                      f'pre reward: {decay_reward:>5.2f}')


class CP0Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self._initialize_weights()

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.normal_(module.bias)


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    model = CP0Model()
    writer = SummaryWriter('tb/reinforce-cartpole-v0')
    reinforce = REINFORCE(env=env,
                          model=model,
                          lr=1e-3,
                          optimizer='adam',
                          gamma=0.95,
                          tbwriter=writer,
                          baseline='mean',
                          deterministic=False)

    reinforce.learning(1000)
    model_base = 'models/'
    if not os.path.exists(model_base):
        os.makedirs(model_base)
        torch.save(model.state_dict(),
                   'models/cartpole-v0-brand-new-baseline.pt')
