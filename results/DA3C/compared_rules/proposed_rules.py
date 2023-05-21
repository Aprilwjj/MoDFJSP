"""
调度规则对比算法
"""
import numpy as np

from environments.SO_DFJSP import SO_DFJSP_Environment
from utilities.Utility_Class import FigGan, MyError, AddData
import os
import random, time

class RulesEnv(SO_DFJSP_Environment):
    def __init__(self, use_instance=True, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.task_rules = [0, 1, 2, 3, 4, 5]  # 工序可选规则列表
        self.machine_rules = [0, 1, 2, 3, 4]  # 机器可选规则列表
        self.rule_dict = {'rule' + str(T) + '_' + str(M): [T, M] for T in self.task_rules for M in self.machine_rules}

    @property
    def rule_random(self):
        return [random.choice(self.task_rules), random.choice(self.machine_rules)]

# 测试环境
if __name__ == '__main__':
    # 读取实例的位置
    path_instance = 'D:\Python project\Deep_Reinforcement_Learning_FJSP\data\DA3C'
    path_writer = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results\DA3C/compared_rules/rules.csv'
    file_name_list \
        = ['DDT0.5_M10_S1', 'DDT0.5_M10_S3', 'DDT0.5_M10_S3', 'DDT0.5_M15_S1', 'DDT0.5_M15_S3', 'DDT0.5_M15_S5',
           'DDT0.5_M20_S1', 'DDT0.5_M20_S3', 'DDT0.5_M20_S5', 'DDT1.0_M10_S1', 'DDT1.0_M10_S3', 'DDT1.0_M10_S5',
           'DDT1.0_M15_S1', 'DDT1.0_M15_S3', 'DDT1.0_M15_S5', 'DDT1.0_M20_S1', 'DDT1.0_M20_S3', 'DDT1.0_M20_S5',
           'DDT1.5_M10_S1', 'DDT1.5_M10_S3', 'DDT1.5_M10_S5', 'DDT1.5_M15_S1', 'DDT1.5_M15_S3', 'DDT1.5_M15_S5',
           'DDT1.5_M20_S1', 'DDT1.5_M20_S3', 'DDT1.5_M20_S5']
    # file_name_list = ['DDT0.5_M10_S1', 'DDT1.0_M10_S1']
    # 生成文件的表头
    data_add = AddData(path_writer)  # 添加数据类对象
    heard = ['Instance'] + ['R_' + str(T) + '_' + str(M) + '_' + str(I)
                            for T in [0, 1, 2, 3, 4, 5] for M in [0, 1, 2, 3, 4] for I in ['best', 'mean', 'std']]
    data_add.add_data(heard)
    # 特定算例下循环固定次数
    epoch_number = 10  # 循环次数
    for file_name in file_name_list:  # 文件循环
        env_object = RulesEnv(use_instance=False, path=path_instance, file_name=file_name)  # 定义环境对象
        best_mean_std = [file_name]  # 文件对应的各规则的最优值均值和标准差
        for rule, actions in env_object.rule_dict.items():  # 规则循环
            data = {'epoch': [], 'objective1': []}  # 初始化写入的数据结构
            for n in range(epoch_number):  # 次数循环---输出最优值+平均值+标准差
                time_start = time.time()
                state = env_object.reset()  # 初始化状态
                while not env_object.done:
                    next_state, reward, done = env_object.step(actions)
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