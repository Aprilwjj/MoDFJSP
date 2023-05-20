"""
通用类
"""
import matplotlib.pyplot as plt
import sys
from random import randint, uniform
import numpy as np, copy, csv
# 英文显示问题
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')  # 设置英文字体
# 中文显示问题
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="SimHei.ttf", size=12)  # 指定中文字体和字号
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""异常类"""
class MyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

"""画图类"""
class FigGan():
    def __init__(self, object):
        self.kind_dict = object.kind_dict

    def figure(self):
        for kind, kind_object in self.kind_dict.items():
            for job_object in kind_object.job_arrive_list:
                for task_object in job_object.task_list:
                    machine = task_object.machine
                    plt.barh(machine, task_object.time_end - task_object.time_begin, left=task_object.time_begin,
                             height=0.4)
                    plt.text(task_object.time_begin, machine + 0.4,
                             '%s|%s|%s)' % (task_object.kind, task_object.number, task_object.task),
                             fontdict={'fontsize': 6})
        plt.show()

"""保存训练结果类"""
class SaveResult():
    def __init__(self, algorithm_name, name_csv, data):
        self.name = algorithm_name  # 算法名字
        self.data = data  # 训练数据
        self.name_csv = name_csv  # csv文件名
        self.objective_count = len(self.data) - 1  # 目标个数
        self.file_name = self.name  # 文件名

    def write_csv(self):
        """写入函数"""
        file_path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results'
        os.makedirs(os.path.join(file_path, self.file_name), exist_ok=True)  # 新建实例文件夹
        file_csv = {self.name_csv + '.csv': ['epoch'] + ['objective' + str(i+1) for i in range(self.objective_count)]}
        best_mean_std = {'best': [], 'mean': [], 'std': []}
        for csv_name, header in file_csv.items():
            data_file = os.path.join(file_path, self.file_name, csv_name)
            with open(data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # 写入最优值，均值和标准差
                rows = []
                for key, value in self.data.items():
                    if key != 'epoch':
                        data = np.array(value)
                        mean = np.mean(data)
                        min_value = np.min(data)
                        std_dev = np.std(data)
                        best_mean_std['best'].append(min_value)
                        best_mean_std['mean'].append(mean)
                        best_mean_std['std'].append(std_dev)
                for key, value in best_mean_std.items():
                    rows.append([key] + value)
                writer.writerows(rows)
                # 写入每个周期的数据
                writer.writerow(header)
                rows = []  # 初始化写入数据
                for row in range(len(self.data['epoch'])):
                    value_list = []
                    for key, value in self.data.items():
                        value_list.append(value[row])
                    rows.append(value_list)
                writer.writerows(rows)
        print("写入完成")


# 测试各类
if __name__ == '__main__':
    figure_object = FigGan
    data_training = {'epoch': [1, 2, 3], 'objective1': [2, 8, 4]}
    result_data = SaveResult(algorithm_name='DA3C', name_csv='training', data=data_training)
    result_data.write_csv()

