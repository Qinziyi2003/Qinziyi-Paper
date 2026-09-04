#  【函数库 - GPR : 高斯过程相关的函数库】
#  1. Fun - 01 和 Fun - 04 分别是自编程和 sklearn 的高斯过程函数
#  2. Fun - 02 是生成测试用的数据
#  3. Fun - 05 是归一化相关的函数
#  4. Fun - 06 是评价指标
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
#  不显示GPR不收敛的提示
import warnings
from sklearn.exceptions import ConvergenceWarning


# =====================================================================================================================
#  Fun - 01 : 自编程的高斯过程 类
class GPR_01_CustomModel:
    """ Fun-01 : 自定义的GP类 """
    def __init__(self, optimize=True):
        """ Sub-00 : 模型初始化 """
        self.is_fit = False
        self.train_x, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize
        self.mu = 0
        self.cov = 0

    def sub_01_inference(self, train_x, train_y, test_x):
        """ Sub-01 : GP模型的推断，也即计算 均值向量 和 协方差阵 """
        #  (1) 录入相关的数据
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        #  (2) 进行核函数的超参数优化
        if self.optimize:
            #  a. 最小化 负对数边际似然 项
            res = minimize(self.sub_03_NLL,
                           [self.params["l"], self.params["sigma_f"]],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            #  b. 更新核函数超参数
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]
        #  (3) 计算核矩阵
        Kff = self.sub_02_kernel(self.train_x, self.train_x)  # (N, N)
        Kyy = self.sub_02_kernel(test_x, test_x)  # (k, k)
        Kfy = self.sub_02_kernel(self.train_x, test_x)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_x)))  # (N, N) 求逆
        #  (4) 计算均值和协方差阵
        self.mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        self.cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        self.is_fit = True
        return self.mu, self.cov

    def sub_02_kernel(self, x1, x2):
        """ Fun-02 : RBF核函数 (类似高斯分布) """
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

    def sub_03_NLL(self, params):
        """ Fun-03 : 定义 NLL 负对数边际似然 公式 """
        #  (1) 传入相关参数
        self.params["l"], self.params["sigma_f"] = params[0], params[1]
        #  (2) 计算 输入变量和核矩阵 Kyy
        Kyy = self.sub_02_kernel(self.train_x, self.train_x) + 1e-8 * np.eye(len(self.train_x))
        #  (3) 定义 负对数边际似然 公式
        loss = (0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) +
                0.5 * np.linalg.slogdet(Kyy)[1] +
                0.5 * len(self.train_x) * np.log(2 * np.pi))
        return loss.ravel()
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 02 : 生成测试数据函数类
def GPR_21_data1D(x, noise_sigma=0.0):
    """ Fun2 - Sub1 : 生成原始数据 输入1D """
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()


def GPR_22_data2D(x, noise_sigma=0.0):
    """ Fun2 - Sub2 : 生成原始数据 输入2D """
    x = np.asarray(x)
    y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 03 : 绘制图像的后处理函数类
def GPR_31_draw1D(gpr, train_X, train_y, test_X, test_y, uncertainty):
    """ Fun3 - Sub1 : 生成图像-1D """
    #  0. 重要参数设置
    num_fontsize = 12
    #  1. 数据前处理
    plt.figure()
    #  2. 置信度区间范围
    plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1, label='Confidence Interval')
    #  3. 预测均值曲线
    plt.plot(test_X, test_y, label="predict")
    #  4. 真实点和预测点标注
    plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    # plt.scatter(test_X, test_y, label="test", c="blue", marker="o")
    #  5. 图像后处理
    plt.title("1D GPR visualization : l=%.2f, sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]),
              fontsize=num_fontsize + 2)
    plt.xlabel('x-label', fontsize=num_fontsize)
    plt.ylabel('y-label', fontsize=num_fontsize)
    plt.legend()
    print(f"    The 1D Gaussian process visualization has been successfully generated.")


def GPR_32_draw2D(gpr, data_x, data_y, data_z, train_X, train_y):
    """ Fun3 - Sub2 : 生成图像-1D """
    #  data_x, data_y, data_z : 预测的输入和输出，输入是一维
    #  train_X, train_y : 训练的输入和输出，输入是二维
    #  0. 重要参数设置
    num_fontsize = 12
    #  1. 数据前处理
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #  2. 绘制图像
    #  (1) 表面图 - 预测值曲面
    ax.plot_surface(data_x, data_y, data_z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    #  (2) 散点图 - 标注真实值
    ax.scatter(np.asarray(train_X)[:, 0], np.asarray(train_X)[:, 1], train_y,
               c=train_y, cmap=cm.coolwarm)
    #  (3) 等高面（投影）- 预测曲面的投影
    ax.contourf(data_x, data_y, data_z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
    #  3. 图像后处理
    ax.set_title("2D GPR visualization : l=%.2f, sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]),
                 fontsize=num_fontsize+2)
    ax.set_xlabel('data_x', fontsize=num_fontsize)
    ax.set_ylabel('data_y', fontsize=num_fontsize)
    print(f"    The 2D Gaussian process visualization has been successfully generated.")


def fun_33_draw2D(gpr, data_x, data_y, data_z, train_X, train_y, uncertainty):
    """ Fun3 - Sub3 : 生成图像-2D 包含置信度区间 """
    #  data_x, data_y, data_z : 预测的输入和输出，输入是一维
    #  train_X, train_y : 训练的输入和输出，输入是二维
    #  0. 重要参数设置
    num_fontsize = 12
    #  1. 数据前处理
    fig = plt.figure(figsize=(12, 5))
    #  2. 绘制图像1 - 预测曲面
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    #  (1) 均值曲面
    ax1.plot_surface(data_x, data_y, data_z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    #  (2) 散点图 - 标注真实值
    ax1.scatter(np.asarray(train_X)[:, 0], np.asarray(train_X)[:, 1], train_y,
                c=train_y, cmap=cm.coolwarm, label='Train Points')
    ax1.contourf(data_x, data_y, data_z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
    #  (3) 图像后处理
    ax1.set_title('GPR - Predict Value Mesh', fontsize=num_fontsize + 2)
    ax1.set_xlabel('data_x', fontsize=num_fontsize)
    ax1.set_ylabel('data_y', fontsize=num_fontsize)
    ax1.set_zlabel('Predict Value', fontsize=num_fontsize)
    #  3. 绘制图像2 - 不确定性曲面
    #  (1) 数据前处理
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #  (2) 绘制误差曲面
    ax2.plot_surface(data_x, data_y, uncertainty, cmap='plasma')
    #  (3) 图像后处理
    ax2.set_title(' GPR - Confidence Interval ', fontsize=num_fontsize + 2)
    ax2.set_xlabel('data_x', fontsize=num_fontsize)
    ax2.set_ylabel('data_y', fontsize=num_fontsize)
    ax2.set_zlabel('Error Mesh', fontsize=num_fontsize)
    #  4. 整体图像后处理
    plt.suptitle("2D GP Visualisation : l=%.2f, sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]),
                 fontsize=num_fontsize + 4)
    plt.tight_layout()


def GPR_34_comparison(x_true, y_true, y_hat):
    """ Fun3 - Sub4 : 高斯过程预测数据和原始数据对比曲线图 """
    #  0. 重要参数设置
    num_fontsize = 12
    gp_r2 = fun_63_r2(y_true, y_hat)
    plt.figure()
    plt.plot(x_true, y_true, label="y_true")
    plt.plot(x_true, y_hat, label="y_hat")
    plt.title(f"Actual vs. Predict, R2 = {gp_r2:.4f}", fontsize=num_fontsize + 2)
    plt.xlabel("Strain Index", fontsize=num_fontsize)
    plt.ylabel("Strain Value", fontsize=num_fontsize)
    plt.legend()
    print(f"    GP prediction vs. original data plot has been drawn successfully ; ")


def GPR_35_comparison(x_train, y_train, x_true, y_true, y_hat, uncertainty):
    """ Fun3 - Sub5 : 高斯过程预测数据和原始数据对比曲线图 """
    #  包含 : 对比曲线、散点图和置信度区间
    #  0. 重要参数设置
    num_fontsize = 12
    #  1. 拟合指标计算
    gp_r2 = fun_63_r2(y_true, y_hat)
    #  2. 绘制图像
    plt.figure()
    #  (1) 真实值和预测值的对比曲线图
    plt.plot(x_true, y_true, label="y_true")
    plt.plot(x_true, y_hat, label="y_hat")
    #  (2) 训练数据的散点图
    plt.scatter(x_train, y_train, label="train", c="red", marker="x")
    #  (3) 置信度区间绘制
    plt.fill_between(x_true.ravel(), y_hat + uncertainty, y_hat - uncertainty, alpha=0.1, label='Confidence Interval')
    #  3. 图像后处理
    plt.title(f"Actual vs. Predict, R2 = {gp_r2:.4f}", fontsize=num_fontsize + 2)
    plt.xlabel("Strain Index", fontsize=num_fontsize)
    plt.ylabel("Strain Value", fontsize=num_fontsize)
    plt.legend()
    print(f"    GP prediction vs. original data plot "
          f"with scatter point and confidence bands has been drawn successfully ; ")
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 04 : sklearn 的高斯过程 类
class GPR_04_sklearnModel:
    """ Fun - 04 : 采用 sklearn 建立高斯过程 """

    def __init__(self, optTime = 20, value_bounds = (1e-8, 1e2), optimize = True, index_ignore=True):
        """ Sub-01 : 模型初始化 """
        self.is_fit = False
        self.train_x, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize
        self.mu = 0
        self.cov = 0
        self.optTime = optTime
        self.value_bounds = value_bounds
        if index_ignore:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)  # 屏蔽收敛警告

    def sub_01_inference(self, train_x, train_y, test_x):
        """ Sub-01 : 建立核函数"""
        kernel = (ConstantKernel(constant_value=self.params["sigma_f"], constant_value_bounds=self.value_bounds)
                  * RBF(length_scale=self.params["l"], length_scale_bounds=self.value_bounds))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.optTime)
        #  2. 进行超参数的拟合
        gpr.fit(train_x, train_y)
        p1 = gpr.kernel_.k2.length_scale
        p2 = gpr.kernel_.k1.constant_value
        self.params["l"] = p1
        self.params["sigma_f"] = p2
        #  3. 进行高斯过程预测
        mu, cov = gpr.predict(test_x, return_cov=True)

        return mu, cov
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 05 : 数据归一化相关函数类
def fun_05_standardization(data):
    """ Fun - 05 : 归一化处理（均值方差归一化）"""
    #  (1) 数据预处理
    num_data = data.shape[0]
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    else:
        data = np.array(data)
    #  (2) 计算均值和方差
    data_mean = data.mean(axis=0)               # 均值
    data_std = data.std(axis=0)                 # 方差
    data_mean = data_mean.reshape(1, -1)
    data_std = data_std.reshape(1, -1)
    #  (3) 【保险】检测是否存在0
    temp_zero = np.array(0, dtype=data_std.dtype)
    for i in range(data_std.shape[1]):
        if data_std[0, i] == temp_zero:
            print(f"第 {i} 列方差是 0")
            data_std[0, i] += 1E-18
    #  (4) 进行归一化处理
    data_standardization = (data - data_mean) / data_std
    return data_standardization, [data_mean, data_std]


def fun_52_normSpecified(data, norm_data):
    """ Fun5 - Sub2 : 按照指定的形式归一化 """
    #  (1) 获取指定归一化信息
    data_mean = norm_data[0]
    data_std = norm_data[1]
    #  (2) 进行归一化处理
    data_standardization = (data - data_mean) / data_std
    return data_standardization


def fun_53_AntiStandard(data, info_ms):
    """ Fun5 - Sub3 : 数据反归一化"""
    #  (1) 数据前处理
    data_mean = info_ms[0]
    data_std = info_ms[1]
    #  (2) 数据反归一化处理
    data_origin = (data * data_std) + data_mean
    return data_origin
# ---------------------------------------------------------------------------------------------------------------------


# =====================================================================================================================
#  Fun - 06 : 评价指标函数类
def fun_61_mse(y_true, y_pred):
    """ Fun6 - Sub1 : 评价指标1 mse"""
    #  1. 进行数据格式转化
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = y_true.reshape(-1, 1)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.reshape(-1, 1)
    #  2. 计算损失数值 MSE
    mse = np.mean((y_true - y_pred) ** 2)
    return mse.item()


def fun_62_mae(y_true, y_pred):
    """ Fun6 - Sub2 : 评价指标2 mae"""
    #  1. 进行数据格式转化
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = y_true.reshape(-1, 1)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.reshape(-1, 1)
    #  2. 计算损失数值 MAE
    y_true_max = np.max(np.abs(y_true))
    mae = np.mean(np.abs(y_true - y_pred)/y_true_max)
    return mae.item()


def fun_63_r2(y_true, y_pred):
    """ Fun6 - Sub3 : 评价指标 R²（决定系数） """
    #  1. 数据格式转化
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1, 1)
    else:
        y_true = y_true.reshape(-1, 1)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().reshape(-1, 1)
    else:
        y_pred = y_pred.reshape(-1, 1)
    # 2. 计算 R² 分数
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()
# ---------------------------------------------------------------------------------------------------------------------

