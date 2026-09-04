#  【函数库 - MIGA : 多岛遗传算法】
#  1. 种群：n_island, popsize, num_dimension
#  2. 包含基础优化算法和后处理程序
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from S2_fitValue_SensorOpt import *


# =====================================================================================================================
#  Fun Group - 01~06 : (1)初始化; (2)适应度计算; (3)选择最优个体
#                      (4)遗传三部曲-选择、交叉和变异; (5)迁移; (6)停止策略
#  I. 种群初始化
def mg_01_initial(popsize, num_dimension, n_island):
    """ Fun - 01 : 生成初始化种群（浮点数编码） """
    #  初始通过 rand 随机生成[0,1)张量，后续再解码操作
    #  1. 生成初始数组 ([0,1]范围)
    pop = torch.rand((n_island, popsize, num_dimension))
    return pop


def mg_02_fitValue(pop, call_fun, scope, index_show=False):
    """ Fun - 02 : 计算种群的适应度 """
    #  1. 对原始数据进行解码
    #  解码包含两个部分: 一个是MIGA的映射范围 ; 二是可能的后续不同任务（例: 转化成node编号），二一般不体现
    decode_pop = pop * (scope[1] - scope[0]) + scope[0]
    #  2. 计算适应度信息值
    #  (1) 预先创建空白数组
    fit_value = torch.ones((pop.shape[0], pop.shape[1]))
    #  (2) 适应度信息计算
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            #  a. 展示不同个体信息
            if index_show:
                print(f"    No ({i+1}, {j+1:>2d}) :", end="")
            #  b. 单个个体适应度信息计算
            temp_individual = decode_pop[i, j, :].reshape(1, -1)
            temp_objValue = call_fun(temp_individual)
            #  c. 进行拼接，存储信息
            fit_value[i, j] = temp_objValue
            #  torch.cat([fit_value, temp_objValue.reshape(1, -1)], dim=0)
    #  3. 将小于0的数值转化成很大数值
    #  保护措施，目前大部分任务是最小化
    fit_value[fit_value < 0] = 10000
    return fit_value


def mg_21_fitValue_Sensor(pop, call_fun, scope, index_show=False, index_valid=False):
    """ Fun - 02 : 计算种群的适应度 """
    #  1. 对原始数据进行解码
    #  解码包含两个部分: 一个是MIGA的映射范围 ; 二是可能的后续不同任务（例: 转化成node编号），二一般不体现
    decode_pop = pop * (scope[1] - scope[0]) + scope[0]
    #  2. 计算适应度信息值
    #  (1) 预先创建空白数组
    fit_value = torch.ones((pop.shape[0], pop.shape[1]))
    r2_nnAll = torch.zeros((pop.shape[0], pop.shape[1]))
    r2_GP = torch.zeros((pop.shape[0], pop.shape[1]))
    #  (2) 适应度信息计算
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            #  a. 展示不同个体信息
            if index_show:
                print(f"    No ({i+1}, {j+1:>2d}) :", end="")
            #  b. 单个个体适应度信息计算
            temp_individual = decode_pop[i, j, :].reshape(1, -1)
            temp_objValue = call_fun(temp_individual)
            #  c. 存储迭代信息，包含 适应度、整体R2、GPR指标R2
            if index_valid:
                fit_value[i, j] = temp_objValue[0]
                r2_nnAll[i, j] = temp_objValue[1]
                r2_GP[i, j] = temp_objValue[2]
            else:
                fit_value[i, j] = temp_objValue
            #  torch.cat([fit_value, temp_objValue.reshape(1, -1)], dim=0)
    #  3. 将小于0的数值转化成很大数值
    #  保护措施，目前大部分任务是最小化
    fit_value[fit_value < 0] = 10000
    if index_valid:
        return fit_value, r2_nnAll, r2_GP
    else:
        return fit_value


#  III. 筛选每代最优个体
def mg_03_tracking(pop, fitValue, data_history):
    """ Fun - 03 : 筛选每一代最优个体并保存 """
    #  存储数据类型解释 :
    #  fitHistory 存储适应度函数，数据类型是张量，是 [num_iteration, num_island+1];
    #  indHistory 存储最优的个体，数据类型是list, 也是类似  [num_iteration, num_island+1];
    #  1. 数据前处理
    num_island, popsize = fitValue.size()
    data_fitHistory = data_history[0]
    data_indHistory = data_history[1]
    #  2. 存储数据的初始化
    temp_fitHistory = torch.zeros((1, num_island))
    temp_indHistory = []
    #  3. 选择最优的个体
    for i in range(num_island):
        temp_fitValue = fitValue[i, :]
        min_value, min_index = torch.min(temp_fitValue, dim=0)
        temp_fitHistory[0, i] = min_value
        temp_indHistory.append(pop[i, min_index, :])
    #  4. 计算每次迭代中最优个体
    min_value, min_index = torch.min(temp_fitHistory, dim=1)
    temp_fitHistory = torch.cat([temp_fitHistory, min_value.reshape(1, -1)], dim=1)
    temp_indHistory.append(temp_indHistory[min_index])
    #  5. 进行数据拼接，存储历代最优信息
    data_indHistory.append(temp_indHistory)
    data_fitHistory = torch.cat([data_fitHistory, temp_fitHistory])
    data_bestHistory = [data_fitHistory, data_indHistory]
    return data_bestHistory


def mg_31_tracking_Sensor(pop, fitValue, r2_nnAll, r2_GP, data_history):
    """ Fun - 03 : 筛选每一代最优个体并保存 """
    #  存储数据类型解释 :
    #  fitHistory 存储适应度函数，数据类型是张量，是 [num_iteration, num_island+1];
    #  indHistory 存储最优的个体，数据类型是list, 也是类似  [num_iteration, num_island+1];
    #  1. 数据前处理
    num_island, popsize = fitValue.size()
    data_fitHistory = data_history[0]
    data_indHistory = data_history[1]
    data_r2_nnAll = data_history[2]
    data_r2_GP = data_history[3]
    #  2. 存储数据的初始化
    temp_fitHistory = torch.zeros((1, num_island))
    temp_indHistory = []
    temp_r2_nnAll = torch.zeros((1, num_island))
    temp_r2_GP = torch.zeros((1, num_island))
    #  3. 选择最优的个体
    for i in range(num_island):
        temp_fitValue = fitValue[i, :]
        temp1_r2_nnAll = r2_nnAll[i, :]
        temp1_r2_GP = r2_GP[i, :]
        min_value, min_index = torch.min(temp_fitValue, dim=0)
        temp_fitHistory[0, i] = min_value
        temp_indHistory.append(pop[i, min_index, :])
        temp_r2_nnAll[0, i] = temp1_r2_nnAll[min_index]
        temp_r2_GP[0, i] = temp1_r2_GP[min_index]
    #  4. 计算每次迭代中最优个体
    min_value, min_index = torch.min(temp_fitHistory, dim=1)
    temp_fitHistory = torch.cat([temp_fitHistory, min_value.reshape(1, -1)], dim=1)
    temp_indHistory.append(temp_indHistory[min_index])
    temp_r2_nnAll = torch.cat([temp_r2_nnAll, temp_r2_nnAll[:, min_index]], dim=1)
    temp_r2_GP = torch.cat([temp_r2_GP, temp_r2_GP[:, min_index]], dim=1)
    #  5. 进行数据拼接，存储历代最优信息
    data_indHistory.append(temp_indHistory)
    data_fitHistory = torch.cat([data_fitHistory, temp_fitHistory])
    data_r2_GP = torch.cat([data_r2_GP, temp_r2_GP])
    data_r2_nnAll = torch.cat([data_r2_nnAll, temp_r2_nnAll])
    data_bestHistory = [data_fitHistory, data_indHistory, data_r2_nnAll, data_r2_GP]
    return data_bestHistory


#  IV. 产生子代数组
def mg_411_select(pop, fitValue):
    """ Fun4 - Sub01 : 选择操作 (单岛局部-轮盘赌选择操作) """
    #  1. 生成概率累加向量
    #  初步生成概率累加数组
    prob_fit = torch.cumsum(fitValue, dim=0)
    #  转化到 [0, 1] 的范畴
    total_fit = torch.sum(fitValue)
    probFit_norm = prob_fit / total_fit
    #  2. 生成概率数组
    #  概率数组是判断落于累加概率区域之中
    popsize = fitValue.shape[0]
    prob_rand = torch.sort(torch.rand(popsize)).values
    #  3. 进行循环判断
    index_cumsum = 0
    index_rand = 0
    new_pop = torch.empty((0, pop.size(1)))
    new_fitValue = torch.empty((0, 1))
    while index_rand <= (popsize - 1):
        if (prob_rand[index_rand]) < probFit_norm[index_cumsum]:
            new_pop = torch.cat((new_pop, pop[index_cumsum, :].reshape(1, -1)), dim=0)
            new_fitValue = torch.cat((new_fitValue, fitValue[index_cumsum].reshape(1, -1)), dim=0)
            index_rand = index_rand + 1
        else:
            index_cumsum += 1
    return new_pop, new_fitValue


def mg_41_select(pop, fitValue):
    """ Fun4 - Sub01 : 选择操作 (整体操作)"""
    #  1. 数据前处理
    num_island, popsize = fitValue.size()
    new_pop = pop.clone()
    new_fitValue = fitValue.clone()
    #  2. 对每个岛进行选择操作
    for i in range(num_island):
        #  【注意是最小值操作】
        #  (1) 分割数据，取对数
        temp_fitValue = 1 / fitValue[i, :]
        temp_pop = pop[i, :, :]
        #  (2) 执行选择操作
        temp_pop, temp_fitValue = mg_411_select(temp_pop, temp_fitValue)
        #  (3) 拼接数组，存储数据
        new_fitValue[i, :] = 1 / temp_fitValue.reshape(1, -1)
        new_pop[i, :, :] = temp_pop
    return new_pop, new_fitValue


def mg_421_crossover(pop, pc):
    """ Fun4 - Sub21 : 交叉操作 (单岛局部 - 单点交叉) """
    #  单点交叉策略
    #  1. 数据前处理
    popsize, num_dimension = pop.size()
    #  2. 对种群进行随机排序
    perm = torch.randperm(popsize)
    pop = pop[perm, :]
    newpop = pop.clone()
    #  3. 进行交叉操作
    for i in range(0, popsize, 2):
        if random.random() < pc:
            #  选择合适的交叉位置，避免在边缘位置
            if num_dimension <= 2:
                random_numbers = random.sample(range(0, num_dimension), 1)[0]
            else:
                random_numbers = random.sample(range(1, num_dimension-1), 1)[0]
            #  执行单点交叉操作
            newpop[i, random_numbers:] = pop[i + 1, random_numbers:]
            newpop[i + 1, random_numbers:] = pop[i, random_numbers:]
    return newpop


def mg_422_crossover(pop, pc):
    """ Fun3 - Sub22 : 交叉操作 (单岛局部 - 双点交叉) """


def mg_423_SBX1(parent1, parent2, eta = 20, lower_bound = 0, upper_bound = 1):
    """ Fun4 - Sub23 : 交叉操作 (单个个体 - SBX交叉) """
    #  1. 生成随机数
    u = torch.rand_like(parent1)
    #  2. 计算 beta 数值
    beta = torch.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    #  3. 执行 SBX 交叉操作
    offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    #  4. 进行边界限制操作
    offspring1 = torch.clip(offspring1, lower_bound, upper_bound)
    offspring2 = torch.clip(offspring2, lower_bound, upper_bound)
    return offspring1, offspring2


def mg_424_SBX2(pop, pc):
    """ Fun4 - Sub24 : 交叉操作 (单岛局部 - SBX交叉) """
    #  1. 数据前处理
    popsize, num_dimension = pop.size()
    #  2. 对种群进行随机排序
    perm = torch.randperm(popsize)
    pop = pop[perm, :]
    newpop = pop.clone()
    #  3. 进行交叉操作
    for i in range(0, popsize, 2):
        if random.random() < pc:
            newpop[i, :], newpop[i + 1, :] = mg_423_SBX1(pop[i, :], pop[i+1, :])
    return newpop


def mg_42_crossover(pop, pc):
    """ Fun3 - Sub02 : 交叉操作 (整体操作) """
    num_island, popsize, num_dimension = pop.size()
    new_pop = pop.clone()
    for i in range(num_island):
        temp_pop = pop[i, :, :]
        temp_pop = mg_424_SBX2(temp_pop, pc)
        # temp_pop = mg_421_crossover(temp_pop, pc)
        new_pop[i, :, :] = temp_pop
    return new_pop


def mg_431_mutation(pop, pm):
    """ Fun4 - Sub31 : 变异操作 (局部 - 单点变异)"""
    #  1. 数据前处理
    popsize, num_dimension = pop.size()
    newpop = pop.clone()
    #  2. 进行变异操作
    for i in range(popsize):
        if random.random() < pm:
            #  (1) 选择变异位置
            #  选择合适的变异位置，避免在边缘位置
            if num_dimension <= 2:
                random_numbers = random.sample(range(0, num_dimension), 1)[0]
            else:
                random_numbers = random.sample(range(1, num_dimension - 1), 1)[0]
            #  (2) 执行变异操作
            newpop[i, random_numbers] = torch.rand(1)
    return newpop


def mg_432_poly(pop, pm, eta = 20, lower_bound = 0, upper_bound = 1):
    """ Fun4 - Sub32 : 变异操作 (局部 - 多项式变异)"""
    #  1. 数据前处理
    new_pop = pop.clone()
    #  2. 变异参数设定
    u = torch.rand_like(new_pop)
    delta = torch.where(
        u < 0.5,
        (2 * u) ** (1 / (eta + 1)) - 1,
        1 - (2 * (1 - u)) ** (1 / (eta + 1))
    )
    #  3. 执行多项式变异操作
    for i in range(pop.size(0)):
        if random.random() < pm:
            for j in range(pop.size(1)):
                new_pop[i, j] += delta[i, j] * (upper_bound - lower_bound)
                new_pop[i, j] = torch.clamp(new_pop[i, j], lower_bound, upper_bound)
    return new_pop


def mg_43_mutation(pop, pm):
    """ Fun4 - Sub03 : 变异操作 (整体操作)"""
    #  1. 数据前处理
    num_island, popsize, num_dimension = pop.size()
    new_pop = pop.clone()
    #  2. 对每一个岛执行变异操作
    for i in range(num_island):
        temp_pop = pop[i, :, :]
        temp_pop = mg_432_poly(temp_pop, pm)
        # temp_pop = mg_431_mutation(temp_pop, pm)
        new_pop[i, :, :] = temp_pop
    return new_pop


def mg_441_elitism(pop_parent, pop_child, fit_value, num_elite, index_show=False):
    """ Fun4 - Sub41 : 精英保留策略：保留最优的个体 """
    #  1. 获取前几个最小值
    #  获取适应度的排序索引（从小到大排序）
    sorted_values, sorted_indices = torch.sort(fit_value)
    #  提取适应度最好的个体索引
    elite_indices = sorted_indices[:num_elite]
    #  2. 返回精英个体
    elite_ind = pop_parent[elite_indices, :]
    #  3. 将子代随机个体替换成亲代最优
    random_numbers = random.sample(range(0, pop_parent.size(0)), num_elite)
    new_popChild = pop_child.clone()
    for i in range(num_elite):
        new_popChild[random_numbers[i], :] = elite_ind[i, :]
    if index_show:
        elite_indices_np = elite_indices.cpu().numpy() + 1
        sorted_values_np = sorted_values[:num_elite].cpu().numpy()
        # 使用 join + 列表推导式格式化数组 - 输出 string
        formatted_values = ", ".join([f"{x:.2f}" for x in sorted_values_np])
        print(f"    The elite is {elite_indices_np} with [{formatted_values}] ; ")
    return new_popChild


def mg_44_elitism(pop_parent, pop_child, fit_value, num_elite, index_show=False):
    """ Fun4 - Sub04 : 变异操作 (整体操作)"""
    #  1. 数据前处理
    num_island, popsize, num_dimension = pop_parent.size()
    new_pop = pop_child.clone()
    #  2. 对每一个岛执行变异操作
    for i in range(num_island):
        print(f"    Island - {i + 1} :", end='')
        new_pop[i, :, :] = mg_441_elitism(pop_parent[i, :, :], pop_child[i, :, :], fit_value[i, :],
                                          num_elite, index_show=index_show)
    return new_pop


def mg_04_reproduction(pop, fitValue, pc, pm, num_elite, index_show=False):
    """ Fun - 03 : 遗传三部曲，产生子代过程 """
    pop1, fitValue1 = mg_41_select(pop, fitValue)
    pop2 = mg_42_crossover(pop1, pc)
    pop3 = mg_43_mutation(pop2, pm)
    pop4 = mg_44_elitism(pop, pop3, fitValue, num_elite, index_show=index_show)
    return pop4


#  VI. 种群迁移
def mg_51_migration(pop1, pop2, migration_ratio):
    """ Fun - 51 : 完全随机的种群迁移操作 """
    #  1. 数据前处理
    popsize = pop1.size(0)
    migration_num = round(migration_ratio * popsize)
    #  2. 完全随机乱序操作
    #  生成两个随机乱序数组
    perm1 = torch.randperm(popsize)
    perm2 = torch.randperm(popsize)
    #  对两个种群进行乱序操作
    new_pop1 = pop1[perm1, :]
    new_pop2 = pop2[perm2, :]
    #  3. 进行交换操作
    temp_ind = new_pop1[0:migration_num, :].clone()
    new_pop1[0:migration_num, :] = new_pop2[0:migration_num, :]
    new_pop2[0:migration_num:, :] = temp_ind
    return new_pop1, new_pop2


def mg_05_migration(pop, migration_ratio, current_iter, num_iterations):
    """ Fun - 05 : 进行种群间的迁移操作 """
    #  1. 数据前处理
    num_island, popsize, num_dimension = pop.size()
    #  2. 执行岛间迁移策略
    if current_iter >= num_iterations//4:
        for i in range(num_island - 1):
            for j in range(i + 1, num_island):
                pop[i, :, :], pop[j, :, :] = mg_51_migration(pop[i, :, :], pop[j, :, :], migration_ratio)
    return pop


#  V. 停止准则


# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 07 : 绘制适应度数值的迭代图
def mg_07_IterationPlot(data_history):
    """ Fun - 07 : 绘制迭代信息的图像 """
    #  0. 重要参数设定
    num_fontsize = 12
    #  1. 生成绘图数据
    data_y = data_history[0][:, -1]
    data_x = range(len(data_y))
    #  2. 绘制图像
    plt.figure()
    plt.plot(data_x, data_y, linestyle='-', color='#4472C4', markersize=5)
    #  3. 图像后处理
    plt.xlabel("迭代次数", fontsize=num_fontsize)
    plt.ylabel("适应度值", fontsize=num_fontsize)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(min(data_y) - 1, max(data_y) + 1)  # 适应度范围控制
    plt.tight_layout()  # 自动调整子图参数，避免标签被截断
    print(f"    Iteration curve plotted successfully.")


def mg_71_IterationPlot(data_history):
    """Fun7 - Sub1 : 绘制 R² 迭代曲线，不绘制适应度值"""

    # 0. 重要参数设定
    num_fontsize = 12

    # 1. 生成绘图数据
    data_y1_r2NN = data_history[2][:, -1]
    data_y2_r2GPR = data_history[3][:, -1]
    data_x = range(len(data_y1_r2NN))

    # 2. 绘制图像
    fig, ax = plt.subplots()

    color_bar = ["#FFBB78", "#98DF8A"]

    ax.plot(
        data_x,
        data_y1_r2NN,
        color=color_bar[0],
        label='R² (RNN)',
        linestyle='--',
        linewidth=2
    )

    ax.plot(
        data_x,
        data_y2_r2GPR,
        color=color_bar[1],
        label='R² (GPR)',
        linestyle='-.',
        linewidth=2
    )

    # 3. 坐标轴设置
    ax.set_xlabel('迭代次数', fontsize=num_fontsize)
    ax.set_ylabel('R²', fontsize=num_fontsize)

    ax.tick_params(axis='both', labelsize=num_fontsize)

    # 4. 图例和网格
    ax.legend(loc='upper right', fontsize=num_fontsize)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. 标题和布局

    plt.tight_layout()
    plt.show()

    print("    R² curves of GPR and NN plotted successfully.")
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
def mg_08_storage(data_bestHistory, scope, filename):
    """" Fun - 08 : 存储解码后的最优个体 """
    #  1. 提取数据
    data_indHistory = data_bestHistory[1]  # 类型:list
    #  物理意义 : 迭代信息、岛的数目
    optimal_sensor_single = data_indHistory[-1][-1]  # 类型 Tensor[n, ]
    #  2. 对数据进行解码
    optimal_sensor_single = optimal_sensor_single * (scope[1] - scope[0]) + scope[0]
    optimal_sensor_node = sensor_01_decode(optimal_sensor_single)
    best_sensor = np.unique(optimal_sensor_node)
    #  3. 最优适应度信息
    best_fitValue = 1 - data_bestHistory[0][-1, -1] / 10000
    #  存储最优个体和适应度信息
    data_comparison = {'optimal_sensor_node': optimal_sensor_node,
                       'best_sensor': best_sensor,
                       'best_fitValue': best_fitValue}
    torch.save(data_comparison, filename)
    print(f"    The best individuals and fitvalue has been saved at {filename} ; ")
# ---------------------------------------------------------------------------------------------------------------------


