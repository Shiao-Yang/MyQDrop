import torch
import torch.nn as nn
import torch.nn.init as init


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data     # 卷积运算过程中的权重
    y_mean = bn_module.running_mean  # 此 batch 计算出的均值μ
    y_var = bn_module.running_var    # 此 batch 计算出的方差σ²
    safe_std = torch.sqrt(y_var + bn_module.eps)  # 标准差
    w_view = (conv_module.out_channels, 1, 1, 1)  # 规范维度
    if bn_module.affine:  # 启用仿射
        weight = w * (bn_module.weight / safe_std).view(w_view)  # bn_module.weight / safe_std = gamma', gamma = bn.weight
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2

# 设置bn的参数
def reset_bn(module: nn.BatchNorm2d):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1-module.eps)
        # we do not reset numer of tracked batches here
        # self.num_batches_tracked.zero_()
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)

# 把模型中所有bn折叠
def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):  # 如果之前的module是Conv2d或者Linear
            fold_bn_into_conv(prev, m)  # fold bn
            # set the bn module to straight through
            setattr(model, n, StraightThrough())  # 由于bn已经fold, 不需要原bn模块, 原模块设置为st
        elif is_absorbing(m):  # 如果当前模块是Conv2d or Linear, 记录当前模块
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)  # 如果都不是, 从头开始搜索当前模块, 执行fold bn操作
    return prev


def search_fold_and_reset_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # reset_bn(m)
        else:
            search_fold_and_reset_bn(m)
        prev = m

