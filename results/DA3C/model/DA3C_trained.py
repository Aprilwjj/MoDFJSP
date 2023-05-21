"""
加载训练好的模型并跑算例
"""
from environments.SO_DFJSP import SO_DFJSP_Environment
from utilities.Utility_Class import FigGan, MyError, AddData
from utilities.Utility_Functions import create_actor_distribution
import os
import random, time, torch
import torch.nn.functional as F
from torch import nn
import numpy as np

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

class PickAction():
    def __init__(self, actions_size, action_types):
        self.actions_size = actions_size
        self.action_types = action_types

    def pick_action(self, policy, state):
        """贪婪的选择动作"""
        state = torch.from_numpy(state).float().unsqueeze(0)  # 状态转为tensor类型
        actor_output = policy.forward(state)
        if policy.name == "task_policy":
            action_size = self.actions_size[0]
        else:
            action_size = self.actions_size[1]
        # 动作分布实例
        action_distribution = create_actor_distribution(self.action_types, actor_output, action_size)  # 动作分布实例
        action = action_distribution.sample().cpu().numpy()  # 采样一个动作
        action = action[0]
        return action

# 测试环境
if __name__ == '__main__':
    # 读取实例的位置
    path_instance = 'D:\Python project\Deep_Reinforcement_Learning_FJSP\data\DA3C'
    path_writer = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results\DA3C/model/DA3C.csv'
    # 实例化网络
    actor_task_model = TaskPolicyNet(input_size_1=20, hidden_size=200, hidden_layer_1=3, output_size_1=6)
    actor_machine_model = MachinePolicyNet(input_size_2=21, hidden_size=200, hidden_layer_2=3, output_size_2=5)
    # 加载训练好的模型参数
    actor_task_model.load_state_dict(torch.load('actor_net_task.ckpt'))
    actor_machine_model.load_state_dict(torch.load('actor_net_machine.ckpt'))
    # 文件列表
    file_name_list \
        = ['DDT0.5_M10_S1', 'DDT0.5_M10_S3', 'DDT0.5_M10_S5', 'DDT0.5_M15_S1', 'DDT0.5_M15_S3', 'DDT0.5_M15_S5',
           'DDT0.5_M20_S1', 'DDT0.5_M20_S3', 'DDT0.5_M20_S5', 'DDT1.0_M10_S1', 'DDT1.0_M10_S3', 'DDT1.0_M10_S5',
           'DDT1.0_M15_S1', 'DDT1.0_M15_S3', 'DDT1.0_M15_S5', 'DDT1.0_M20_S1', 'DDT1.0_M20_S3', 'DDT1.0_M20_S5',
           'DDT1.5_M10_S1', 'DDT1.5_M10_S3', 'DDT1.5_M10_S5', 'DDT1.5_M15_S1', 'DDT1.5_M15_S3', 'DDT1.5_M15_S5',
           'DDT1.5_M20_S1', 'DDT1.5_M20_S3', 'DDT1.5_M20_S5']
    # file_name_list = ['DDT0.5_M10_S1']  # 测试算例
    data_add = AddData(path_writer)  # 添加数据类对象
    # 特定算例下循环固定次数
    epoch_number = 10  # 循环次数
    for file_name in file_name_list:  # 文件循环
        env_object = SO_DFJSP_Environment(use_instance=False, path=path_instance, file_name=file_name)  # 定义环境对象
        pick_action_object = PickAction(env_object.actions_size, env_object.action_types)
        data = {'epoch': [], 'objective1': []}  # 初始化写入的数据结构
        best_mean_std = [file_name]  # 文件对应的各规则的最优值均值和标准差
        for n in range(epoch_number):  # 次数循环---输出最优值+平均值+标准差
            # 运行一次循环
            time_start = time.time()  # 循环开始时间
            state = env_object.reset()  # 初始化状态
            while not env_object.done:
                action_task = pick_action_object.pick_action(actor_task_model, state)
                state_add = np.append(state, action_task)  # 带选择的工序规则信息的状态
                action_machine = pick_action_object.pick_action(actor_machine_model, state_add)
                actions = np.array([action_task, action_machine])  # 二维离散动作
                next_state, reward, done = env_object.step(actions)
                state = next_state
            # 保存数据
            data['epoch'].append(n)  # 写入代数
            data['objective1'].append(env_object.delay_time_sum)  # 写入目标值
        # 计算该规则的最优值，均值和标准差
        data_array = np.array(data['objective1'])
        best_mean_std.extend([np.mean(data_array), np.min(data_array), np.std(data_array)])
        # 写入文件
        data_add.add_data(best_mean_std)

    # 画甘特图
    # figure_object = FigGan(env_object)
    # figure_object.figure()

