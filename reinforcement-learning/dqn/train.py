# -*- coding: utf-8 -*-
from math import ceil

import gym
import numpy as np
import torch
from torch import nn
from torch import optim

from env import SkipframeWrapper


class EpisodeData(object):

    def __init__(self):
        self.fields = [
            'states', 'actions', 'rewards', 'dones', 'log_probs', 'next_states'
        ]
        for f in self.fields:
            setattr(self, f, [])
        self.total_rewards = 0

    def add_record(self,
                   state,
                   action,
                   reward,
                   done,
                   log_prob=None,
                   next_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.total_rewards += reward

    def get_states(self):
        return np.array(self.states)

    def get_actions(self):
        return np.array(self.actions)

    def steps(self):
        return len(self.states)

    def calc_qs(self, pre_model, gamma):
        next_states = torch.tensor(np.array(self.next_states)).float()
        next_qs = pre_model(next_states).max(dim=-1).values
        masks = torch.tensor(np.array(self.dones) == 0)

        rewards = torch.tensor(np.array(self.rewards)).view(-1)
        qs = rewards + gamma * next_qs * masks

        return qs.detach().float()


class DQN(object):

    def __init__(self,
                 env,
                 model,
                 lr=1e-5,
                 optimizer='adam',
                 device='cpu',
                 deterministic=False,
                 gamma=0.95,
                 n_replays=4,
                 batch_size=200,
                 model_kwargs=None,
                 exploring=None,
                 n_trained_times=1,
                 n_buffers=32,
                 model_prefix="dqn"):
        self.env = env
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.device = device
        self.deterministic = deterministic
        self.gamma = gamma
        self.n_replays = n_replays
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        self.exploring = exploring
        self.n_trained_times = n_trained_times

        if self.model_kwargs:
            self.pre_model = self.model.__class__(**self.model_kwargs)
        else:
            self.pre_model = self.model.__class__()

        self.data_buffer = []
        self.n_buffers = n_buffers
        self.model_prefix = model_prefix

        self.copy_model()

    def gen_epoch_data(self, n_steps=1024, exploring=0., done_penalty=0):
        state = self.env.reset()
        done = False
        epoch_data = EpisodeData()

        self.model.eval()
        steps = 0

        for _ in range(n_steps):
            steps += 1

            qs = self.model(torch.tensor(state[np.newaxis, :]).float())

            if exploring and np.random.rand() <= exploring:
                action = self.env.action_space.sample()
            else:
                action = qs[0].argmax().item()

            next_state, reward, done, _ = self.env.step(int(action))
            if done and done_penalty:
                reward -= done_penalty

            epoch_data.add_record(state,
                                  action,
                                  reward,
                                  1 if done else 0,
                                  next_state=next_state)
            state = next_state

            if done:
                state = self.env.reset()

        return epoch_data

    def get_exploring(self, need_exploring=False, mexp=0.1):
        if need_exploring:
            return max(mexp, self.n_trained_times**(-0.5))
        if isinstance(self.exploring, float):
            return self.exploring
        elif self.exploring == 'quadratic_decrease':
            return max(0.01, self.n_trained_times**(-0.5))

        return 0.01

    def copy_model(self):
        self.pre_model.load_state_dict(self.model.state_dict())
        self.pre_model.eval()

    def train(self, epoch_data):
        total_loss = 0.
        qs = epoch_data.calc_qs(self.pre_model, gamma=0.95).to(self.device)
        states = torch.tensor(epoch_data.get_states()).float().to(self.device)
        actions = torch.tensor(epoch_data.get_actions()[:, np.newaxis]).to(
            self.device)

        n_batches = ceil(len(epoch_data.states) / self.batch_size)
        indices = torch.randperm(len(epoch_data.states)).to(self.device)
        for b in range(n_batches):
            batch_indices = indices[b * self.batch_size:(b + 1) *
                                    self.batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_qs = qs[batch_indices]

            qs_pred = self.model(batch_states).gather(1,
                                                      batch_actions).view(-1)
            loss_func = nn.MSELoss()
            loss = loss_func(batch_qs, qs_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_batches

    def learning(self, n_epoches=100, n_steps=1024):
        self.model.train()

        max_reward = -10000.
        decay_reward = 0
        decay = 0.95

        for n in range(n_epoches):
            # generate new data
            new_data = self.gen_epoch_data(n_steps=n_steps,
                                           exploring=self.get_exploring()
                                           if not self.deterministic else 0.)
            self.data_buffer.insert(0, new_data)
            if len(self.data_buffer) > self.n_buffers:
                self.data_buffer = self.data_buffer[:self.n_buffers]

            # training
            for data in self.data_buffer[::-1]:
                loss = self.train(data)

            # update static model
            self.copy_model()

            # show training information
            decay_reward = new_data.total_rewards if decay_reward == 0 else (
                decay_reward * decay + new_data.total_rewards * (1 - decay))

            if max_reward < decay_reward:
                max_reward = decay_reward
                torch.save(self.model.state_dict(),
                           f'./models/{self.model_prefix}-success-v{n}.pt')

            if n % 10 == 0:
                print(
                    f'round: {n:>3d} | loss: {loss:>5.3f} | '
                    f'pre reward: {decay_reward:>5.2f}',
                    flush=True)


class ModuleInitMixin:

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.05)
                nn.init.normal_(module.bias, 0, 0.1)


class DARModel(ModuleInitMixin, nn.Module):

    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.device = device
        self._initialize_weights()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()

        x = x.to(self.device)
        return self.fc(x)


if __name__ == '__main__':

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    env = SkipframeWrapper(env=gym.make('DemonAttack-ram-v0'), n_max_nops=60)
    model = DARModel(device)
    dqn = DQN(env=env,
              model=model,
              exploring='quadratic_decrease',
              device=device,
              lr=5e-4,
              gamma=0.95,
              model_prefix='dqndar',
              batch_size=25)

    model = model.to(device)

    import os
    try:
        model_base = './models/'
        if not os.path.exists(model_base):
            os.makedirs(model_base)

        dqn.learning(11, 256)

        torch.save(model.state_dict(), './models/dqn-dar-train-final.pt')
    except Exception as e:
        torch.save(model.state_dict(), './models/dqn-dar-train-except.pt')
        raise
