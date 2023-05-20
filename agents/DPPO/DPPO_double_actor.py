"""
双重异步优势演员评论家算法
三个网络：工序策略网络+机器策略网络+评论家网络
"""
import copy
import random
import time
import numpy as np
import torch
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.Utility_Functions import create_actor_distribution, normalise_rewards
from utilities.Parallel_Experience_Generator import Parallel_Experience_Generator
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Config import Config
from environments.SO_DFJSP import SO_DFJSP_Environment
import torch.nn.functional as F
from torch import nn
from visdom import Visdom

# 监控训练过程
window_name = 'Double Actor state(0)+action(0)+reward(1)'
vis = Visdom()
win = window_name
title = window_name
vis.line(X=[0], Y=[0], win=win, opts=dict(title=title, xlabel='epoch', ylable='total_delay_time',
                                          font=dict(family='Times New Roman')))

# 构建工序策略网络类
class TaskPolicyNet(nn.Module):
    def __init__(self, input_size_1, hidden_size, hidden_layer_1, output_size_1):
        super(TaskPolicyNet, self).__init__()
        self.name = "task_policy"
        # 定义工序策略网络输入层
        self.layers_1 = nn.ModuleList([nn.Linear(input_size_1, hidden_size), nn.ReLU()])
        # 定义工序策略网络隐藏层
        for i in range(hidden_layer_1 - 1):
            self.layers_1.append(nn.Linear(hidden_size, hidden_size))
            self.layers_1.append(nn.ReLU())
        # 定义工序策略网络输出层
        self.layers_1.append(nn.Linear(hidden_size, output_size_1))

    def forward(self, x):
        for layer in self.layers_1:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x

# 构建机器策略网络类
class MachinePolicyNet(nn.Module):
    def __init__(self, input_size_2, hidden_size, hidden_layer_2, output_size_2):
        super(MachinePolicyNet, self).__init__()
        self.name = "machine_policy"
        # 定义机器策略网络输入层
        self.layers_2 = nn.ModuleList([nn.Linear(input_size_2, hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer_2 - 1):
            self.layers_2.append(nn.Linear(hidden_size, hidden_size))
            self.layers_2.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers_2.append(nn.Linear(hidden_size, output_size_2))

    def forward(self, x):
        for layer in self.layers_2:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x

# 构建评论家网络
class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(CriticNet, self).__init__()
        # 定义评论家网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义评论家网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义评论家网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DPPO(Base_Agent, Config):
    """双重近端策略优化算法"""
    agent_name = 'DPPO'
    def __init__(self):
        Base_Agent.__init__(self)
        Config.__init__(self)
        self.policy_output_size = self.calculate_policy_output_size()
        self.policy_task_new = TaskPolicyNet(input_size_1=20, hidden_size=200, hidden_layer_1=3, output_size_1=6)
        self.policy_old = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old.load_state_dict(copy.deepcopy(self.policy_task_new.state_dict()))
        self.policy_new_optimizer = optim.Adam(self.policy_task_new.parameters(), lr=self.hyper_parameters["learning_rate"],
                                               eps=1e-4)
        self.episode_number = 0
        self.many_episode_states = []  # 状态
        self.many_episode_actions = []  # 动作
        self.many_episode_rewards = []  # 回报
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_task_new, self.config.seed,
                                                                  self.hyper_parameters, self.action_size)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)

    def calculate_policy_output_size(self):
        """Initialises the policies"""
        if self.action_types == "DISCRETE":
            return self.action_size
        elif self.action_types == "CONTINUOUS":
            return self.action_size * 2  # Because we need 1 parameter for mean and 1 for std of distribution

    def step(self):
        """Runs a step for the PPO agent"""
        exploration_epsilon = self.exploration_strategy.get_updated_epsilon_exploration(
            {"episode_number": self.episode_number})
        self.many_episode_states, self.many_episode_actions, self.many_episode_rewards = self.experience_generator.play_n_episodes(
            self.hyper_parameters["episodes_per_learning_round"], exploration_epsilon)
        self.episode_number += self.hyper_parameters["episodes_per_learning_round"]
        self.policy_learn()  # 循环learning_iterations_per_round次更新策略
        self.update_learning_rate(self.hyper_parameters["learning_rate"], self.policy_new_optimizer)  # 更新学习率
        self.equalise_policies()  # 更新旧策略网络参数

    def policy_learn(self):
        """A learning iteration for the policy"""
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.hyper_parameters["normalise_rewards"]:
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        for _ in range(self.hyper_parameters["learning_iterations_per_round"]):
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)

    def calculate_all_discounted_returns(self):
        """
        Calculates the cumulative discounted return for each episode which we will then use in a learning iteration
        """
        all_discounted_returns = []
        for episode in range(len(self.many_episode_states)):
            discounted_returns = [0]
            for ix in range(len(self.many_episode_states[episode])):
                return_value = self.many_episode_rewards[episode][-(ix + 1)] + self.hyper_parameters["discount_rate"] * \
                               discounted_returns[-1]
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns

    def calculate_all_ratio_of_policy_probabilities(self):
        """
        For each action calculates the ratio of the probability that the new policy would have picked the action vs.
        the probability the old policy would have picked it. This will then be used to inform the loss
        """
        all_states = [state for states in self.many_episode_states for state in states]
        all_actions = [[action] if self.action_types == "DISCRETE" else action for actions in self.many_episode_actions
                       for action in actions]
        all_states = torch.stack([torch.Tensor(states).float().to(self.device) for states in all_states])

        all_actions = torch.stack([torch.Tensor(actions).float().to(self.device) for actions in all_actions])
        all_actions = all_actions.view(-1, len(all_states))

        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_task_new, all_states,
                                                                                     all_actions)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_old, all_states,
                                                                                     all_actions)
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (
                    torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """Calculates the log probability of an action occuring given a policy and starting state"""
        policy_output = policy.forward(states).to(self.device)
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):
        """Calculates the PPO loss"""
        all_ratio_of_policy_probabilities = torch.squeeze(torch.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities, min=-sys.maxsize,
                                                        max=sys.maxsize)
        all_discounted_returns = torch.tensor(all_discounted_returns).to(all_ratio_of_policy_probabilities)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * self.clamp_probability_ratio(
            all_ratio_of_policy_probabilities)
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        loss = -torch.mean(loss)
        return loss

    def clamp_probability_ratio(self, value):
        """Clamps a value between a certain range determined by hyperparameter clip epsilon"""
        return torch.clamp(input=value, min=1.0 - self.hyper_parameters["clip_epsilon"],
                           max=1.0 + self.hyper_parameters["clip_epsilon"])

    def take_policy_new_optimisation_step(self, loss):
        """Takes an optimisation step for the new policy"""
        self.policy_new_optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.policy_task_new.parameters(), self.hyper_parameters[
            "gradient_clipping_norm"])  # clip gradients to help stabilise training
        self.policy_new_optimizer.step()  # this applies the gradients

    def equalise_policies(self):
        """将旧策略的参数设置为与新策略的参数相等"""
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_task_new.parameters()):
            old_param.data.copy_(new_param.data)








