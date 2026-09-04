#  【程序 - B2 : 不规则波浪下的传感器优化】
#  1. 采用 GPR+NN 的方式解决传感器位置可变的问题
#  2. z91 存储传感器优化所需的数据
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from functools import partial
import matplotlib.pyplot as plt
from S1_MIGA import *
from S2_fitValue_SensorOpt import *

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

print(f"\nProgram - B2 : Sensor Optimization of irRegular wave : ")
#  I. 加载原始数据集
print(f"\nThe Part I:")
#  1. 原始数据集
filename_temp = "z91_SensorOpt_irRegData.pth"
data_all = torch.load(filename_temp, weights_only=False)
data_features = data_all['data_features']
data_labels = data_all['data_labels']
data_CF = data_all['data_CF']
net = data_all['net']
print(net)
print(f"    The data and net has been loaded, and the shape of data is {data_features.shape} and {data_labels.shape}.")
#  2. 转化数据为 Tensor.float
data_features = data_features.float()
data_labels = data_labels.float()
#  3. 计算数据维度
num_feature = data_features.shape[1]
num_label = data_labels.shape[1]
num_data = data_features.shape[0]
#  4. 计算峰值时刻索引
data_CF = torch.mean(data_CF, dim=1)
time_index = torch.argmax(data_CF)
print(f"    The selected time index is {time_index + 1} ; ")


#  II. 基本参数设定
print(f"\nThe Part II : ")
#  0. 重要参数
index_valid = True
num_dimension = 225   # 输入变量的维度
pop_size = 6         # 种群数目
num_islands = 2       # 岛的个数
num_iterations = 200  # 迁移迭代参数
#  1. 输入参数
scope = [0, num_feature-1]  # 输入数据的取值范围
#  2. 迭代相关参数
# num_iterations = 200  # 迁移迭代参数
# pop_size = 10  # 种群数目
pc = 0.7
pm = 0.2  # 变异交叉概率
# num_islands = 4  # 岛的个数
migration_ratio = 0.1  # 迁移率
num_elite = 1  # 精英保留数目
print(f"    The parameters has been loaded, "
      f"with num_dimension={num_dimension}, pop_size={pop_size}, num_islands={num_islands} ; ")


#  III. 数据前处理
print(f"\nThe Part III : ")
#  1. 种群初始化
pop = mg_01_initial(pop_size, num_dimension, num_islands)
#  2. 存储数据初始化
data1_fitHistory = torch.empty((0, num_islands + 1))
data2_indHistory = []
data3_r2_nnAll = torch.empty((0, num_islands + 1))
data4_r2_GP = torch.empty((0, num_islands + 1))
data_history = [data1_fitHistory, data2_indHistory, data3_r2_nnAll, data4_r2_GP]
#  3. 适应度函数初始化（GP&NN拟合误差）
call_fun = partial(sensor_02_fitting,
                   data_features=data_features, data_labels=data_labels,
                   net=net, time_index=time_index, index_valid=index_valid)
print(f"    The data has been initialized. ")


#  IV. 开始GA迭代计算
print(f"\nThe Part IV:")
for i in range(num_iterations):
    # (1) 适应度计算
    fitValue, r2_nnAll, r2_GP = mg_21_fitValue_Sensor(pop, call_fun, scope, index_show=True, index_valid=index_valid)
    # (2) 筛选操作
    data_history = mg_31_tracking_Sensor(pop, fitValue, r2_nnAll, r2_GP, data_history)

    # (3) 产生新的子代个体
    newpop1 = mg_04_reproduction(pop, fitValue, pc, pm, num_elite, index_show=True)
    # (4) 迁移操作
    newpop2 = mg_05_migration(newpop1, migration_ratio, i, num_islands)

    # (5) 执行迭代必要操作
    #  更新种群
    pop = newpop2
    #  输出迭代信息
    print(f"Iteration {i + 1}: Best fitness = {data_history[0][-1, -1]:.2f}")


#  IV. 数据后处理以及绘制图像
#  1. 存储最优传感器方案以及对应整体适应度信息
filename = "x1_OptimalSensor_ind.pth"
mg_08_storage(data_history, scope, filename)
#  2. 适应度图像
mg_07_IterationPlot(data_history)
mg_71_IterationPlot(data_history)

plt.show()
print(f"\nThe Sensor Optimization is over!\n")

