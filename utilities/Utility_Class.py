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

class MyError(Exception):
    """异常类"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FigGan():
    """画图类"""
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

class AddData():
    """添加训练数据"""
    def __init__(self, path_file_name):
        self.file_name = path_file_name

    def add_data(self, data):
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

# 测试各类
if __name__ == '__main__':
    # 添加训练数据
    data_training = [1, 2, 3]
    path_file_name = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results/DA3C/training.csv'
    result_data = AddData(path_file_name)
    result_data.add_data(data_training)

