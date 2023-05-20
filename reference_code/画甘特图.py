"""
甘特图子函数
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # mcolors.TABLEAU_COLORS可以得到一个字典，可以选择TABLEAU_COLORS,CSS4_COLORS等颜色组
import matplotlib.font_manager as fm


# 加工完成的工序 [0工件，1工步，2加工机器，3开始时间，4完成时间]

def gan_te_tu(msg, machine_amount):
    myfont = fm.FontProperties(fname='..\Deng.ttf')
    colors = list(mcolors.XKCD_COLORS.keys())  # 颜色变化
    """
    画主甘特图
    :param msg: [0工件，1工步，2加工机器，3开始时间，4完成时间]
    """
    plt.figure(1)
    ax = plt.gca()
    [ax.spines[i].set_visible(False) for i in ["top", "right"]]

    for i in range(len(msg)):
        plt.barh(msg[i][2], msg[i][4] - msg[i][3], left=msg[i][3], height=0.9, color=mcolors.XKCD_COLORS[colors[msg[i][0]]])
        plt.text(msg[i][3] + 0.1, msg[i][2] - 0.35,
                 '工件-%s\n工步-%s' % (msg[i][0], msg[i][1]), color="black", size=6, fontproperties=myfont)

    machine_list = list(range(machine_amount))
    plt.yticks(machine_list, [i for i in machine_list])
    plt.ylabel('Machine Number')
    plt.xlabel('Process Time /h')
    plt.savefig('.\GTT.png')
    plt.show()


import pandas as pd

import pandas as pd

df = pd.read_excel(r'C:\Users\11367\OneDrive\毕业论文\图\5_实际案例_堆场车辆调度问题\实际案例.xlsx')
df = df[['车辆编号', '组件类型', '区域', '对应站点', '开始时间', '结束时间']]
df['区域'] = df['区域']
df['对应站点'] = df['对应站点']
print(df)
msg = []    # param msg: [0工件，1工步，2加工机器，3开始时间，4完成时间]
for index, row in df.iterrows():
    msg.append([int(row['车辆编号']),int(row['组件类型']),int(row['对应站点']),row['开始时间'],row['结束时间']])

machine_amount = 12
gan_te_tu(msg, machine_amount)