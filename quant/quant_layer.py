import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# straight-through estimator: 用于量化过程的求导
# 替代原来的backward函数
class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

# straight through estimator: 用于量化网络中。forward时, 把float量化, backward时, 用量化参数的梯度直接更新原float参数, 同时会做截断处理
# 通常ste使用的函数可以自己编写, 即自定义backward方式
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

# 求lp范数作为损失函数
def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

# 执行非对称量化的类, 即量化器
class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization 量化比特数
    :param symmetric: if True, the zero_point should always be 0 是否为对称量化
    :param channel_wise: if True, compute scale and zero_point in each channel 是否为逐通道量化
    :param scale_method: determines the quantization scale and zero point TODO: 决定量化的scale和zero point?
    :param leaf_param: if True, the param is a leaf param 是否是叶子参数, 叶子参数使用ema来设置scale
    :param prob: for qdrop; qdrop中量化节点被替换的概率
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        if self.sym:
            raise NotImplementedError
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits  # 量化后值的最大值
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        # ema得到的值在图像上更加平缓光滑，抖动性更小，不会因为某次的异常取值而使得滑动平均值波动很大, 设置scale很合适
        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''mse params'''
        self.scale_method = 'mse'
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    # 设置是否手动初始化 TODO: 详细作用
    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    # 更新量化范围, 该函数是ema的具体实现, 0.9为设置的decay, 根据输入的min和max, 更新对应的最大最小值
    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    # 前向传播
    def forward(self, x: torch.Tensor):
        # 如果不是手动初始化, 执行自动初始化. delta: 量化比例scale; zero_point: 零点
        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point   # Q = R/S + Z
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)    # 限制范围在[0, 2^bit-1]
        x_dequant = (x_quant - self.zero_point) * self.delta  # R = (Q-Z) * S
        # 执行QDrop
        if self.is_training and self.prob < 1.0:
            # torch.where(condition, a, b): 按照一定的规则合并两个tensor类型, 若满足条件condition, 选择a, 否则选择b
            # torch.rand_like(x): 产生一个size和x相同, 从[0,1)上的均匀分布随机取值的tensor
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)  # 随机drop掉一些x_dequant的值, 用原浮点类型的x的对应值替换
        else:
            x_ans = x_dequant
        return x_ans

    # 计算lp范数作为损失函数, p=2时即为l2范数
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)  # l2范数
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)  # 逐通道求损失函数, 把第一维及其之后的维数拼接起来, 组成多个一维tensor, 每个代表一个通道的数据
            return y.mean(1)  # dim=1, 按行求平均值，返回的形状是(行数，1), 即每个通道的平均值

    # 计算scale和zero_point
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        # torch.zeros_like:生成和括号内变量维度一致的全0的tensor
        # TODO: 为什么要限制最小值和最大值范围
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))  # 限制最小值 <= 0
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))  # 限制最大值 >= 0

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)  # 计算scale
        scale = torch.max(scale, self.eps)  # 因为有除操作, 防止scale=0
        zero_point = quant_min - torch.round(min_val_neg / scale)   # zero_point = Q_max-R_max/scale = Q_min-R_min/scale
        zero_point = torch.clamp(zero_point, quant_min, quant_max)  # 限制zero_point在quant_min和quant_max之间
        return scale, zero_point

    # 执行量化操作
    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)    # 返回一维的长度和x的维度相同的全1的list
            new_shape[0] = x.shape[0]         # 第一个元素大小和x的第一维维度相同, 即通道数
            # reshape delta 和 zero_point 到 new_shape 的形状, 方便逐通道计算
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        # x_int = round(x/scale) + z
        # x_Q = clamp(x, 0, n_levels-1)
        # x_dequant = (x_Q - z) * scale
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    # 搜索最大最小值
    '''
    当输入的值在0的左侧和右侧都有分布时, 使用该函数寻找最优的最大值和最小值, 即对原有的float数做截断
    '''
    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)  # 逐通道求损失函数, 把第一维及其之后的维数拼接起来, 组成多个一维tensor, 每个代表一个通道的数据
            x_min, x_max = torch._aminmax(y, 1)  # 求第二维的最大最小值
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))  # 限制最大值 >= 0
            x_min = torch.min(x_min, torch.zeros_like(x_min))  # 限制最小值 <= 0
        else:
            x_min, x_max = torch._aminmax(x)  # 否则求所有值的最大最小值
        xrange = x_max - x_min  # float scale
        best_score = torch.zeros_like(x_min) + (1e+10)  # 得分, 可以理解为损失函数值
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        '''
        Q = R/S + Z
        R = (Q - Z) * S
        Z = Q_max - R_max / S = Q_min - R_min / S
        R_max = Q_max * S - Z * S; R_min = Q_min * S - Z * S
        除以self.num是为了枚举浮点值范围, 用来求出最佳的截断位置
        做截断的目的是舍弃掉某些值过大的浮点值, 使得结果更好, 量化能更好的应对异常值(过大和过小值)
        '''
        for i in range(1, self.num + 1):  #
            tmp_min = torch.zeros_like(x_min)  # tmp_min ≈ Q_min * S
            tmp_max = xrange / self.num * i    # tmp_max ≈ Q_max * S   R
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp: 枚举零点, 不同零点带来不同的量化后值的分布. 零点配合量化范围就能求出浮点数最大最小值
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta  # R_min = Q_min * S - Z * S
                new_max = tmp_max - zp * tmp_delta  # R_max = Q_max * S - Z * S
                x_q = self.quantize(x, new_max, new_min)  # 执行量化, 得到dequant tensor
                score = self.lp_loss(x, x_q, 2.4)  # 求量化损失
                best_min = torch.where(score < best_score, new_min, best_min)  # 根据量化损失, 更新最小值
                best_max = torch.where(score < best_score, new_max, best_max)  # 根据量化损失, 更新最大值
                best_score = torch.min(best_score, score)
        return best_min, best_max

    # 搜索最大最小值
    '''
    当输入的值全部在0的左侧或右侧, 或采用对称量化时, 使用该函数寻找最优的最大值和最小值, 即对原有的float数做截断
    由R = (Q - Z) * S可知, 此处默认zero_point就是int值范围的最左端, 即0
    和对称量化本质类似, 扩大了源数据范围, 会削弱int8的表示能力
    '''
    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres  # 如果输入全为正, 最小值为0, 否则为-thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres   # 如果输入全为负, 最大值为0, 否则为thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    # 获取浮点数最大最小值
    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.sym:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:  # 如果是叶子参数, 还要使用ema更新最大最小值
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    # 全通道初始化scale和zero_point
    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    # 逐通道(可选)初始化scale和zero_point
    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)  # 得到一个全1的list, 长度是x_clone的shape, 由于调用该函数前, tensor一般都已经reshape为二维的, 实际为[1,1]
            new_shape[0] = x_clone.shape[0]       # 第一位设为通道数, 即[x_clone.shape[0], 1]
            delta = delta.reshape(new_shape)      # reshape为[x_clone.shape[0], 1]
            zero_point = zero_point.reshape(new_shape)  # reshape为[x_clone.shape[0], 1]
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    # 设置量化bit数
    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )

# 专门用来替换普通卷积块和普通线性层的对应量化版模块, 可以执行普通卷积(全连接)或者量化版卷积(全连接)
class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()
        # fwd_kwargs: 存储变长参数, 为关键字参数, 是一个字典
        # fwd_func: 存储函数操作

        # 如果是conv2d, 传递相应参数, 函数操作为conv2d
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        # 如果是linear, 无需传递参数, 函数操作为linear
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight  # 模型的权重
        self.org_weight = org_module.weight.data.clone()  # 存储原模型的权重
        if org_module.bias is not None:
            self.bias = org_module.bias  # 模型的bias
            self.org_bias = org_module.bias.data.clone()  # 存储原模型的bias
        else:
            self.bias = None
            self.org_bias = None

        # 默认不使用量化
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()  # 激活函数为StraightThrough
        self.ignore_reconstruction = False  # TODO
        self.disable_act_quant = disable_act_quant  # TODO

    def forward(self, input: torch.Tensor):
        # 如果使用权重量化, 执行量化权重操作
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        # 否则, 使用原权重
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)  # 执行卷积或全连接操作
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        out = self.activation_function(out)  # 求激活值
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    # 设置是否量化weight以及是否量化activation
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
        )
