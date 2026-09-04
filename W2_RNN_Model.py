import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
#  函数库02 - 定义 RNN 神经网络模型


#  I. 创建用类定义的RNN模型
class w2_01_rnnClass(nn.Module):
    """ Fun - 01 : 循环神经网络模型 """
    # 用类定义的神经网络模型
    def __init__(self, num1_hiddens_rnn, num2_hiddens_ann, num_feature, num_label,
                 type_init, index_activation="Tanh", **kwargs):
        """ Fun1 - Sub1 : RNN类初始化 """
        #  1. 继承父类用法
        super().__init__(**kwargs)
        #  2. 神经网络层数和节点数目定义
        num1_hiddens_rnn.insert(0, num_feature)
        num2_hiddens_ann.insert(0, num1_hiddens_rnn[-1])
        num2_hiddens_ann.append(num_label)
        self.num1_hiddens_rnn = num1_hiddens_rnn
        self.num2_hiddens_ann = num2_hiddens_ann
        self.rnn = nn.ModuleList()
        self.ann = nn.ModuleList()
        #  2. 神经网络模型定义
        #  Part I - RNN部分
        for i in range(len(num1_hiddens_rnn)-1):
            temp_rnn = nn.RNN(num1_hiddens_rnn[i], num1_hiddens_rnn[i+1], batch_first=True)
            # temp_rnn = nn.RNN(num1_hiddens_rnn[i], num1_hiddens_rnn[i + 1])  # 【vocab 模式】
            self.rnn.append(temp_rnn)
        #  Part II - 线性层部分
        for i in range(len(num2_hiddens_ann)-1):
            #  最后一个是输出层的大小，因此不必遍历
            #  (1) 定义预备的线性层和激活层
            temp_linear = nn.Linear(num2_hiddens_ann[i], num2_hiddens_ann[i + 1])
            if index_activation == "ReLU":
                temp_activation = nn.ReLU()
            else:
                temp_activation = nn.Tanh()
            #  (2) 向 ANN 中添加层
            if i != len(num2_hiddens_ann) - 2:
                #  不是最后一层的情况
                self.ann.append(temp_linear)
                self.ann.append(temp_activation)
            else:
                #  是最后一层的情况
                self.ann.append(temp_linear)
        #  3. 参数初始化
        self._init_weights(type_init)

    def _init_weights(self, type_init="normal"):
        """ Fun1 - Sub2 : 神经网络参数初始化函数 """
        #  对 RNN 的权重进行 Xavier 初始化
        #  1. 定义初始化的方式
        if type_init == "xavier_normal":
            init_method = nn.init.xavier_normal_
        elif type_init == "xavier_uniform":
            init_method = nn.init.xavier_uniform_
        else:
            init_method = lambda tensor: nn.init.normal_(tensor, mean=0.0, std=0.01)
        #  2. 执行 Part I - RNN 部分 参数初始化
        for i in range(len(self.rnn)):
            for name, param in self.rnn[i].named_parameters():
                if 'weight_ih' in name:  # 输入到隐藏层的权重
                    init_method(param)
                elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重
                    init_method(param)
                elif 'bias' in name:  # 偏置初始化为零
                    init.zeros_(param)
        #  3. 执行 Part II - ANN 部分 参数初始化
        for i in range(len(self.ann)):
            if isinstance(self.ann[i], nn.Linear):
                init_method(self.ann[i].weight)
                init.zeros_(self.ann[i].bias)

    def forward(self, inputs, state):
        """ Fun1 - Sub3 : 正向传播过程 """
        # 【vocab 模式】
        # num_vocab = self.num1_hiddens_rnn[0]
        # inputs = F.one_hot(inputs.T.long(), num_vocab)
        # inputs = inputs.clone().detach().float()
        #  1. RNN 部分
        input_rnn = inputs
        output_state = []
        for i in range(len(self.rnn)):
            temp_output, temp_state = self.rnn[i](input_rnn, state[i])
            # temp_output, temp_state = self.rnn[i](input_rnn, state)  # 【vocab 模式】
            input_rnn = temp_output
            output_state.append(temp_state)
        # output_rnn = input_rnn[:, -1, :]  # 【A1模式】
        output_rnn = input_rnn.reshape(-1, input_rnn.size(-1))  # 【A2&A3模式】t步输入预测t步输出
        # output_rnn = input_rnn.reshape(-1, input_rnn.shape[-1])  # 【vocab 模式】
        #  2. ANN部分
        input_ann = output_rnn
        for i in range(len(self.ann)):
            temp_output = self.ann[i](input_ann)
            input_ann = temp_output
        output_ann = input_ann
        return output_ann, output_state

    def begin_state(self, device, batch_size=1):
        """ Fun1 - Sub4 : 隐状态初始化"""
        hidden_state = []
        for i in range(len(self.rnn)):
            temp_hidden_state = torch.zeros(1, batch_size, self.rnn[i].hidden_size).to(dtype=torch.float32)
            temp_hidden_state = temp_hidden_state.to(device)
            hidden_state.append(temp_hidden_state)
        return hidden_state
        # return temp_hidden_state  # 【vocab 模式】
