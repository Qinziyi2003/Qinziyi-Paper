import math
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils import data
# from d2l import torch as d2l
from torch.utils.data import Dataset


#  I. 数据累加器
class w3_01_Accumulator:
    """ Fun - 01 : 不断累加指定数据 """
    def __init__(self, n):
        #  1. 初始化空白数组
        self.data = [0.0] * n
        self.n = n

    def add(self, *args):
        #  2. 将新的数据进行累加
        #  先组合成键对值，在一一加和
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        #  3. 成原重置始的空白数组
        self.data = [0.0] * self.n

    def __getitem__(self, idx):
        #  4. 以索引方式访问类数据
        return self.data[idx]


#  II. 评价指标
def w3_21_r2(y_true, y_pred):
    """ Fun2 - Sub1 : 评价指标 R2 """
    #  (1) 残差平方和 SS_res
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    ss_res = ((y_pred - y_true) ** 2).sum(dim=0)
    #  (2) 总平方和 SS_tot（以每个输出维度的均值为基准）
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0)
    #  (3) 加上一个小的 epsilon 防止除以 0
    epsilon = 1e-8
    r2 = 1 - ss_res / (ss_tot + epsilon)
    #  (4) 平均多个输出维度的 R2
    return r2.mean().item()


def w3_22_mse(y_true, y_pred):
    """ Fun2 - Sub2 : 评价指标 mse"""
    loss = nn.MSELoss()
    mse = loss(y_pred, y_true)
    return mse.item()


def w3_23_mae(y_true, y_pred, remain_all=False):
    """ Fun2 - Sub3 : 评价指标 mae"""
    #  1. 计算mae的数值
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    #  2. 所有数值的均值进行计算
    # y_delta = (y_true - y_pred)
    y_delta = (y_true - y_pred) / y_true
    mae_abs = np.abs(y_delta)
    #  3. 要保留数据的格式
    if remain_all:
        mae_final = mae_abs
    else:
        mae_final = mae_abs.mean()
    return mae_final


#  III. 训练函数
def w3_31_gradClipping(net, theta):
    """ Fun3 - Sub1 : 裁剪梯度 """
    #  1. 筛选涉及梯度的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    #  2. 计算相关梯度幅值
    norm2 = sum(torch.sum((p.grad ** 2)) for p in params)
    norm2 = norm2.detach().to(torch.float32)
    norm = torch.sqrt(norm2)
    norm = norm.item()
    #  3. 进行梯度裁剪
    if norm > theta:
        print(f"         The norm is {norm}.")
        for param in params:
            param.grad[:] *= theta / norm


def w3_32_trainEpoch_RNN(net, get_train_iter, loss, updater, split_type, device):
    """ Fun3 - Sub2 : 训练网络一个迭代周期（定义见第8章）"""
    #  1. 数据预处理
    #  (1) 是否采用随机划分方式
    if split_type == "A3":
        use_random_iter = True
    else:
        use_random_iter = False
    #  (2) 参数初始化
    state = None
    metric = w3_01_Accumulator(2)  # 训练损失之和, 处理的数据批次
    #  (3) dataloader 初始化
    train_iter = get_train_iter()
    # train_iter = get_train_iter  # 【vocab 模式】

    #  2. 开启训练模型
    if isinstance(net, torch.nn.Module):
        net.train()

    #  3. 进行单个 epoch 的神经网络训练
    #  (1) 计算初始隐状态
    for X, Y in train_iter:
        if state is None or use_random_iter:
            #  在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(device, batch_size=X.shape[0])
        else:
            # s 有两种情况：
            # (1) 自编程的分段层的隐状态，GRU和原始RNN网络
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                for s in state:
                    s.detach_()
                # state.detach_()  # 【vocab 模式】
            # (2) LSTM 等其他网络还可能特殊一点，是多列表
            else:
                for s1 in state:
                    for s2 in s1:
                        s2.detach_()

        #  (2) 训练神经网络 - 传统五步走策略
        #  计算、损失、清零、梯度、更新
        #  a - 计算
        X, Y = X.to(device), Y.to(device)
        # Y = Y.T.reshape(-1)  # 【vocab 模式】
        y = Y.reshape(-1, Y.shape[-1])
        y_hat, state = net(X, state)
        #  b - 损失
        # loss_train = loss(y_hat, y.long()).mean()  # 【vocab 模式】
        loss_train = loss(y_hat, y).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()  # c - 清零
            loss_train.backward()  # d - 梯度
            w3_31_gradClipping(net, 1)  # 梯度裁剪
            updater.step()  # e - 更新
        else:
            loss_train.backward()  # d - 梯度
            w3_31_gradClipping(net, 1)  # 梯度裁剪
            updater(batch_size=1)  # e - 更新

        #  (3) 信息输出
        num_steps = X.shape[0] * X.shape[1]
        metric.add(loss_train * num_steps, num_steps)

    # return math.exp(metric[0] / metric[1])  # 【vocab 模式】
    return metric[0] / metric[1]


def w3_33_extract(get_data_iter):
    """ Fun3 - Sub3 ：提取迭代器中所有数据 """
    #  1. 按照 list 的格式提取相关数据
    data_iter = get_data_iter()
    list_x, list_y = [], []
    for x, y in data_iter:
        list_x.append(x)
        list_y.append(y)
    #  2. 将 list 数据进行数据拼接成完整 Tensor
    all_x = torch.cat(list_x, dim=0)
    all_y = torch.cat(list_y, dim=0)
    # all_y = all_y.squeeze()
    all_x = all_x.reshape(-1, all_x.shape[-1])
    all_y = all_y.reshape(-1, all_y.shape[-1])
    return all_x, all_y


def w3_03_trainAll_RNN(net, get_train_iter, get_test_iter, loss, updater, num_epochs,
                       num_steps, device, split_type="A2"):
    """ Fun - 03 : 训练模型 """
    #  1. 数据前处理
    # ls_info = torch.zeros(num_epochs, 1)  # 【vocab 模式】
    ls_info = torch.zeros(num_epochs, 5)

    #  2. 单个 epoch 的训练
    for epoch in range(num_epochs):
        train_mse = w3_32_trainEpoch_RNN(net, get_train_iter, loss, updater, split_type, device)

        #  3. 记录损失信息 & 评价指标
        #  (1) 训练集评估
        train_x, train_y = w3_33_extract(get_train_iter)
        train_y_hat = w3_04_predict_RNN(net, train_x, device)
        train_r2 = w3_21_r2(train_y, train_y_hat)
        #  (2) 验证集评估
        valid_x, valid_y = w3_33_extract(get_test_iter)
        valid_y_hat = w3_04_predict_RNN(net, valid_x, device)
        valid_r2 = w3_21_r2(valid_y, valid_y_hat)
        valid_mse = w3_22_mse(valid_y, valid_y_hat)

        #  4. 输出每次迭代的损失信息
        ls_info[epoch, :] = torch.tensor([epoch, train_r2, train_mse, valid_r2, valid_mse])
        print(f"    The {epoch + 1} th train, the train_r2 is {train_r2:.4f}, the valid_r2 is {valid_r2:.4f}...")
        # 【vocab 模式】
        # ls_info[epoch, 0] = train_mse
        # print(f"    The {epoch + 1} th train, the loss is {ls_info[epoch, 0]:.4f} ;")
    return ls_info


def w3_04_predict_RNN(net, x_forecast, device, warmup_data=None, time_steps=1):
    """ Fun - 04 : 在prefix后面生成新字符 """
    #  1. 初始化隐状态
    state = net.begin_state(device, batch_size=1)
    x_forecast = x_forecast.to(device)

    #  2. 使用预热数据初始化隐状态（如果提供了 warmup_data）
    if warmup_data is not None:
        for x_warm in warmup_data:
            _, state = net(x_warm, state)

    #  3. 开启评估模式（如果是 nn.Module）
    if isinstance(net, torch.nn.Module):
        net.eval()

    #  4. 进行预测 - 逐个对 [num_steps, num_features] 进行预测
    y_forecast = []
    with torch.no_grad():
        for step_idx, x_input in enumerate(x_forecast, start=1):
            x_input = x_input.reshape(1, 1, -1)
            # 根据 time_steps 的值决定是否更新隐状态
            if step_idx % time_steps == 0 and time_steps != 0:
                y, state = net(x_input, state)
            else:
                y, _ = net(x_input, state)

            y_forecast.append(y)

    y_forecast = torch.cat(y_forecast, dim=0)
    return y_forecast


def w3_05_params(net, wd):
    """ Fun - 05 : 提取权值和偏置的list """
    #  为接下来 L2 正则化做准备
    #  weight_decay 是对全部的参数施加正则化, 因此需要分块
    #  1. 数据处理
    param_w = []
    param_b = []
    #  2. 提取参数便于后期施加正则化
    #  (1) 提取 RNN 的参数
    for i in range(len(net.rnn)):
        for name, param in net.rnn[i].named_parameters():
            if 'weight' in name:
                param_w.append(param)
            elif 'bias' in name:
                param_b.append(param)
    #  (2) 提取 ANN 中的相关参数
    for i in range(len(net.ann)):
        if isinstance(net.ann[i], nn.Linear):
            param_w.append(net.ann[i].weight)
            param_b.append(net.ann[i].bias)
    #  3. 返回优化器所需的参数
    temp1 = {'params': param_w, 'weight_decay': wd}
    temp2 = {'params': param_b}
    return [temp1, temp2]
