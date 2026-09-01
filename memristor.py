"""
忆阻器模型：实现LTP（长期增强）和LTD（长期抑制）机制
基于HP忆阻器模型和生物神经可塑性原理
"""

import torch
import torch.nn as nn
import numpy as np


class MemristorLTPLTD:
    """
    忆阻器LTP/LTD学习规则
    
    LTP (Long-Term Potentiation): 长期增强 - 前后神经元同时激活时，权重增加
    LTD (Long-Term Depression): 长期抑制 - 前后神经元活动不相关时，权重减少
    """
    
    def __init__(self, learning_rate=0.001, ltp_amplitude=0.01, ltd_amplitude=0.01, 
                 weight_min=-1.0, weight_max=1.0, threshold=0.1):
        """
        参数：
            learning_rate: 学习速率
            ltp_amplitude: LTP强度（权重增强幅度）
            ltd_amplitude: LTD强度（权重减弱幅度）
            weight_min/max: 权重限制范围
            threshold: 激活阈值
        """
        self.learning_rate = learning_rate
        self.ltp_amplitude = ltp_amplitude
        self.ltd_amplitude = ltd_amplitude
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.threshold = threshold
        
    def update_weight(self, weight, pre_activity, post_activity):
        """
        使用LTP/LTD规则更新权重
        
        参数：
            weight: 当前权重
            pre_activity: 前神经元活动（0-1之间）
            post_activity: 后神经元活动（0-1之间）
        
        返回：
            更新后的权重
        """
        # 计算前后神经元活动的乘积（Hebbian相关性）
        correlation = pre_activity * post_activity
        
        # LTP: 相关性高时增强权重
        if correlation > self.threshold:
            dw_ltp = self.ltp_amplitude * correlation * (1 - torch.abs(weight))
            weight = weight + self.learning_rate * dw_ltp
        
        # LTD: 相关性低时减弱权重
        elif correlation < -self.threshold:
            dw_ltd = -self.ltd_amplitude * torch.abs(correlation) * (1 - torch.abs(weight))
            weight = weight + self.learning_rate * dw_ltd
        
        # 权重限制在指定范围内
        weight = torch.clamp(weight, self.weight_min, self.weight_max)
        
        return weight
    
    def batch_update(self, weights, pre_activities, post_activities):
        """
        批量更新权重
        
        参数：
            weights: 权重张量 (batch_size, input_size, output_size)
            pre_activities: 前层活动 (batch_size, input_size)
            post_activities: 后层活动 (batch_size, output_size)
        
        返回：
            更新后的权重
        """
        updated_weights = weights.clone()
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                for k in range(weights.shape[2]):
                    updated_weights[i, j, k] = self.update_weight(
                        weights[i, j, k],
                        pre_activities[i, j],
                        post_activities[i, k]
                    )
        
        return updated_weights


class MemristorLinear(nn.Module):
    """
    集成忆阻器LTP/LTD学习的线性层
    """
    
    def __init__(self, in_features, out_features, memristor_params=None, 
                 use_memristor=True, bias=True):
        """
        参数：
            in_features: 输入特征数
            out_features: 输出特征数
            memristor_params: 忆阻器参数字典
            use_memristor: 是否使用忆阻器学习规则
            bias: 是否使用偏置
        """
        super(MemristorLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_memristor = use_memristor
        
        # 标准线性层参数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # 初始化忆阻器
        if self.use_memristor:
            memristor_params = memristor_params or {}
            self.memristor = MemristorLTPLTD(
                learning_rate=memristor_params.get('learning_rate', 0.001),
                ltp_amplitude=memristor_params.get('ltp_amplitude', 0.01),
                ltd_amplitude=memristor_params.get('ltd_amplitude', 0.01),
                weight_min=memristor_params.get('weight_min', -1.0),
                weight_max=memristor_params.get('weight_max', 1.0),
                threshold=memristor_params.get('threshold', 0.1)
            )
        
        # 记录活动用于学习
        self.pre_activities = None
        self.post_activities = None
    
    def forward(self, x):
        """前向传播"""
        if self.use_memristor:
            # 记录前层活动（归一化到0-1）
            self.pre_activities = torch.sigmoid(x)
        
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        if self.use_memristor:
            # 记录后层活动
            self.post_activities = torch.sigmoid(output)
        
        return output
    
    def memristor_update(self):
        """
        使用忆阻器LTP/LTD规则更新权重
        """
        if not self.use_memristor or self.pre_activities is None:
            return
        
        # 简化版：对平均活动进行更新
        pre_avg = self.pre_activities.mean(dim=0)  # (in_features,)
        post_avg = self.post_activities.mean(dim=0)  # (out_features,)
        
        # 外积计算权重更新
        for i in range(self.out_features):
            for j in range(self.in_features):
                correlation = pre_avg[j] * post_avg[i]
                
                # LTP: 相关性高时增强权重
                if correlation > self.memristor.threshold:
                    dw = self.memristor.ltp_amplitude * correlation * \
                         (1 - torch.abs(self.weight[i, j]))
                    self.weight.data[i, j] += self.memristor.learning_rate * dw
                
                # LTD: 相关性低时减弱权重
                else:
                    dw = -self.memristor.ltd_amplitude * torch.abs(correlation) * \
                         (1 - torch.abs(self.weight[i, j]))
                    self.weight.data[i, j] += self.memristor.learning_rate * dw
                
                # 权重裁剪
                self.weight.data[i, j] = torch.clamp(
                    self.weight.data[i, j],
                    self.memristor.weight_min,
                    self.memristor.weight_max
                )


class MemristorConv2d(nn.Module):
    """
    集成忆阻器LTP/LTD学习的卷积层
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, memristor_params=None, use_memristor=True):
        """
        参数：
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            memristor_params: 忆阻器参数
            use_memristor: 是否使用忆阻器学习规则
        """
        super(MemristorConv2d, self).__init__()
        
        self.use_memristor = use_memristor
        
        # 标准卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding)
        
        # 初始化忆阻器
        if self.use_memristor:
            memristor_params = memristor_params or {}
            self.memristor = MemristorLTPLTD(
                learning_rate=memristor_params.get('learning_rate', 0.0001),
                ltp_amplitude=memristor_params.get('ltp_amplitude', 0.005),
                ltd_amplitude=memristor_params.get('ltd_amplitude', 0.005),
                weight_min=memristor_params.get('weight_min', -1.0),
                weight_max=memristor_params.get('weight_max', 1.0),
                threshold=memristor_params.get('threshold', 0.1)
            )
        
        # 记录活动
        self.input_activity = None
        self.output_activity = None
    
    def forward(self, x):
        """前向传播"""
        if self.use_memristor:
            # 记录输入活动
            self.input_activity = torch.sigmoid(x)
        
        output = self.conv(x)
        
        if self.use_memristor:
            # 记录输出活动
            self.output_activity = torch.sigmoid(output)
        
        return output
    
    def memristor_update(self):
        """
        使用忆阻器LTP/LTD规则更新卷积核权重
        """
        if not self.use_memristor or self.input_activity is None:
            return
        
        # 简化版：使用平均活动进行更新
        input_avg = self.input_activity.mean(dim=(0, 2, 3))  # (in_channels,)
        output_avg = self.output_activity.mean(dim=(0, 2, 3))  # (out_channels,)
        
        # 更新卷积权重
        for out_c in range(self.conv.out_channels):
            for in_c in range(self.conv.in_channels):
                correlation = input_avg[in_c] * output_avg[out_c]
                
                # LTP
                if correlation > self.memristor.threshold:
                    dw = self.memristor.ltp_amplitude * correlation * \
                         (1 - torch.abs(self.conv.weight[out_c, in_c]))
                    self.conv.weight.data[out_c, in_c] += self.memristor.learning_rate * dw
                
                # LTD
                else:
                    dw = -self.memristor.ltd_amplitude * torch.abs(correlation) * \
                         (1 - torch.abs(self.conv.weight[out_c, in_c]))
                    self.conv.weight.data[out_c, in_c] += self.memristor.learning_rate * dw
                
                # 权重裁剪
                self.conv.weight.data[out_c, in_c] = torch.clamp(
                    self.conv.weight.data[out_c, in_c],
                    self.memristor.weight_min,
                    self.memristor.weight_max
                )
