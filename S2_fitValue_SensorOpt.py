#  【函数库 - sensor : 传感器的适应度信息】
#  1.
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from S3_GP import *
from W2_RNN_Model import *
from W3_trainRNN import *


# =====================================================================================================================
#  Fun - 01 : 单个个体的解码
def sensor_01_decode(single):
    """ Fun - 01 : 对原始个体数据进行解码，解码结果是node编号 """
    # 解决原始个体四舍五入后可能存在重复数值的问题
    # 1. 原始四舍五入
    x_rounded = torch.round(single)
    x_rounded = x_rounded.reshape([1, -1])

    # 2. 找出重复值
    unique_vals, counts = torch.unique(x_rounded, return_counts=True)
    duplicates = unique_vals[counts > 1]

    # 3. 修正重复数值
    # (1) 复制以便修改
    x_adjusted = x_rounded.clone()

    # (2) 对重复数据进行更新，增加随机性
    for val in duplicates:
        # a. 找到这个重复值对应的位置
        mask = (x_rounded == val)
        idx = torch.nonzero(mask).squeeze()
        idx = idx[:, 1]

        # b. 原始值中，把这些挑出来
        orig_vals = x_adjusted[0, idx]

        # c. 根据原始值排序
        sorted_idx = torch.argsort(orig_vals)

        # d. 更新数值，即四舍五入的node
        half = len(sorted_idx) // 2
        for i in range(len(sorted_idx)):
            orig_i = idx[sorted_idx[i]]
            if i < half:
                # 向下取整
                x_adjusted[0, orig_i] = torch.floor(x_adjusted[0, orig_i])
            else:
                # 向上取整
                x_adjusted[0, orig_i] = torch.ceil(x_adjusted[0, orig_i])

                # 增加随机性，随机选择一部分进行向下或向上取整
            if torch.rand(1).item() > 0.5:
                x_adjusted[0, orig_i] = torch.floor(x_adjusted[0, orig_i])  # 向下取整
            else:
                x_adjusted[0, orig_i] = torch.ceil(x_adjusted[0, orig_i])  # 向上取整

    # 4. 进行数据格式的转化
    x_adjusted = x_adjusted.cpu().numpy().astype(int)

    # 检查解码后的节点数量，确保它们是多样化的
    unique_nodes = len(set(x_adjusted[0]))  # 计算唯一节点的数量
    if unique_nodes < 5:  # 如果节点太少，可能需要进一步调整策略
        print(f"Warning: Too few unique nodes detected: {unique_nodes}")

    return x_adjusted


# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 02 : 单个个体的适应度函数计算
def sensor_02_fitting(individual, data_features, data_labels, net, time_index,
                      index_sort=True, index_figure=False, index_valid=False):
    """ Fun - 02 : 计算单个个体的适应度值 (外部适应度单个个体即可） """

    # 1. 选取训练所需的 strain 数据
    element_indices = sensor_01_decode(individual)
    time_index = time_index.to(dtype=torch.long)
    strain1_discrete = data_features[time_index, element_indices]  # 选取离散的应变数据
    strain2_peakTime = data_features[time_index, :]  # 选取典型时刻应变数据
    y_true = strain2_peakTime  # 真实值是原始分布的应变数据

    # 2. GP - 离散到全场应变数据的扩充
    num_element = data_features.shape[1]
    train_x = element_indices.reshape(-1, 1)
    train_y = strain1_discrete.reshape(-1, 1).cpu().numpy()
    test_x = np.arange(0, num_element, 1).reshape(-1, 1)

    # 进行数据排序处理
    time1 = time.time()
    if index_sort:
        train_x_sort, test_x_sort, y_true_sort = sensor_41_sort(train_x, train_y, test_x, y_true)
    time2_sort = time.time()

    # 进行数据归一化
    if not index_sort:
        test_x_norm, tsx_mv = fun_05_standardization(test_x)
        train_x_norm = fun_52_normSpecified(train_x, tsx_mv)
    else:
        test_x_norm, tsx_mv = fun_05_standardization(test_x_sort)
        train_x_norm = fun_52_normSpecified(train_x_sort, tsx_mv)

    # 进行高斯过程的推断和预测
    gpr = GPR_04_sklearnModel()
    mu, _ = gpr.sub_01_inference(train_x_norm, train_y, test_x_norm)
    strain2_global = mu.ravel()
    time3_gp = time.time()
    gp_mse = fun_61_mse(y_true, strain2_global)
    gp_mae = fun_62_mae(y_true, strain2_global)
    gp_r2 = fun_63_r2(y_true, strain2_global)

    # 3. NN拟合CF数据
    strain2_global = torch.from_numpy(strain2_global).float().reshape(1, -1)
    if isinstance(net, w2_01_rnnClass):
        device = torch.device("cpu")
        y_hat = w3_04_predict_RNN(net, strain2_global, device)
    elif isinstance(net, nn.Sequential):
        net.eval()
        with torch.no_grad():
            y_hat = net(strain2_global).reshape(1, -1)
    time4_nn = time.time()

    # 4. 计算适应度信息
    y_true = data_labels[time_index, :].reshape(1, -1)
    mse = fun_61_mse(y_true, y_hat)
    r2 = fun_63_r2(y_true, y_hat)
    fitvalue_final = 10 ** 4 * (1 - r2)  # 最终适应度值

    # 5. 输出和调试信息
    print(f"    Single fitness takes {time4_nn - time1:.4f} s; "
          f"FitValue = {fitvalue_final:>11.4f}, r2 = {r2:7.4f}, mse = {mse:.4f}; "
          f"GPR - mse={gp_mse:.4f}, mae={gp_mae:.4f}, r2={gp_r2:>7.4f}; "
          f"with {time3_gp - time2_sort:.4f} s by GP , {time4_nn - time3_gp:.4f} s by NN; ")

    if index_valid:
        return fitvalue_final, r2, gp_r2
    else:
        return fitvalue_final

    # ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 03 : 绘制图像函数类
def sensor_03_figure(train_x, train_y, y_true, y_hat, gp_r2):
    """ Fun - 03 : 展示每次插值的效果 """
    #  包含训练点的标注 和 真实预测值的曲线对比
    #  0. 重要参数设定
    num_fontsize = 12
    #  1. 数据前处理
    #  (1) 图窗定义
    plt.figure()
    #  (2) 计算 单元 的数目
    num_element = y_hat.shape[0]
    #  (3) 生成绘图数据
    test_x = np.arange(0, num_element, 1)
    #  2. 绘制图像
    #  (1) 真实值数据曲线
    plt.plot(test_x, y_true, label="y_true")
    #  (2) 预测值数据曲线
    plt.plot(test_x, y_hat, label="y_hat")
    #  (3) 训练数据的散点图
    plt.scatter(train_x, train_y, label="train", c="red", marker="o")
    #  (4) 图像后处理
    plt.title(f"The single GPR : R2 = {gp_r2:.4f}", fontsize=num_fontsize+2)
    plt.xlabel("Strain Index", fontsize=num_fontsize)
    plt.ylabel("Strain Value", fontsize=num_fontsize)
    plt.legend()
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 04 : 数据排序相关函数类
def sensor_41_sort(train_x, train_y, test_x, y_true, index_check=False):
    """ Fun4 - Sub1 : 在GP插值之前预先对数据进行排序 """
    #  1. 对原始 strain 数据进行排序
    indices_sort = np.argsort(y_true)
    y_true_sort = y_true[indices_sort]
    #  2. 对原始x进行反向映射
    #  按照排序后顺序对原始的x顺序进行更改
    position_map = np.empty_like(indices_sort)
    position_map[indices_sort] = np.arange(len(indices_sort))
    train_x_sort = position_map[train_x]
    test_x_sort = position_map[test_x]
    #  3. 检测数据是否相等
    #  (1) 根据转化后的 train_x 数据在排序后数据提取的 strain
    test_y1 = y_true_sort[train_x_sort].reshape(-1, 1).cpu().numpy()
    #  (2) 原始 train_y 数据
    test_y2 = train_y
    #  (3) 检测是否相等
    if index_check:
        if np.array_equal(test_y1, test_y2):
            print("     The data is equal after sorting transformation.")
        else:
            print("     The data is not equal after sorting transformation.")
    return train_x_sort, test_x_sort, y_true_sort


def sensor_42_AntiSort(y_true, strain2_global):
    """ Fun4 - Sub2 : 对数据进行反排序处理 """
    indices_sort = np.argsort(y_true)
    indices_inverse = np.argsort(indices_sort)
    strain3_AntiSort = strain2_global[indices_inverse]
    return strain3_AntiSort
# ---------------------------------------------------------------------------------------------------------------------

