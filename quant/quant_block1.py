import torch.nn as nn
from .quant_layer import QuantModule, UniformAffineQuantizer
from models.resnet import BasicBlock, Bottleneck
from models.regnet import ResBottleneckBlock
from models.mobilenetv2 import InvertedResidual
from models.mnasnet import _InvertedResidual
from ldm.modules.diffusionmodules.openaimodel import (
    TimestepBlock,
    ResBlock,
    )

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from ldm.modules.attention import (
    default,
    exists,
    CrossAttention,
)

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False

    # 递归设置是否使用block的量化版的激活值和权重
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

# override BasicBlock in resnet-18 and resnet-34
class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        # 将conv1和conv2设置为量化版, TODO: ? conv1的激活函数保留原relu1
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        # 下采样与原BasicBlock相同
        # 此处下采样就是要输出的通道数, 使得conv2的输出通道数和输入通道数相同
        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        self.activation_function = basic_block.relu2  # 激活函数保留为原basicblock的relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)  # 激活值量化器

    def forward(self, x):
        # 如果有下采样器, 则进行下采样
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        # 是否使用激活值量化
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        # conv1, conv2, conv3都执行量化, 激活函数暂时保留
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        # 如果下采样器不为空, 执行下采样
        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3
        # 激活值量化器
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        # 与BasicBlock同理
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

# notice: 继承的是BaseQuantBlock 而不是原模块
class QuantResBlock(BaseQuantBlock):
    # 外部信息输入, 但没保存为内部变量self, 如何访问?
    def __init__(self, resblock: ResBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.quant_in_layers = nn.Sequential(
            resblock.in_layers[0],
            resblock.in_layers[1],
            QuantModule(resblock.in_layers[2], weight_quant_params, act_quant_params),
        )

        # self.emb_layers里有if 怎么处理?
        # config传入原模块，直接调原来的
        self.quant_emb_layers = nn.Sequential(
            resblock.emb_layers[0],
            QuantModule(resblock.emb_layers[1], weight_quant_params, act_quant_params),
        )

        self.quant_out_layers = nn.Sequential(
            resblock.out_layers[0],
            resblock.out_layers[1],
            resblock.out_layers[2],
            QuantModule(resblock.out_layers[3],, weight_quant_params, act_quant_params),
        )

        # nn.Identity()
        if resblock.out_channels == resblock.channels:
            self.quant_skip_connection = resblock.skip_connection
        # use_conv或者两者都不是时, skip_connection都是conv_nd
        else:
            self.quant_skip_connection = QuantModule(resblock.skip_connection, weight_quant_params, act_quant_params)

        # 应该是self还是resblock? weight_quant_params? TODO
        def forward(self, x, emb):
            return checkpoint(
                self._forward, (x, emb), resblock.parameters(), resblock.use_checkpoint
            )

        def _forward(self, x, emb):
            if resblock.updown:
                quant_in_rest, quant_in_conv = self.quant_in_layers[:-1], self.quant_in_layers[-1]
                h = quant_in_rest(x)

                # upd没动
                h = resblock.h_upd(h)
                x = resblock.x_upd(x)

                h = quant_in_conv(h)
            else:
                h = self.quant_in_layers(x)

            emb_out = self.quant_emb_layers(emb).type(h.dtype)
            # 对齐没动
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

            if resblock.use_scale_shift_norm:
                out_norm, out_rest = self.quant_out_layers[0], self.quant_out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.quant_out_layers(h)

            out = self.quant_skip_connection(x) + h
            if self.use_act_quant:
                out = self.act_quantizer(out)
            return out

class QuantCrossAttention(BaseQuantBlock):
    def __init__(self, crossattention: CrossAttention, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.quant_to_q = QuantModule(crossattention.to_q,, weight_quant_params, act_quant_params)
        self.quant_to_k = QuantModule(crossattention.to_k,, weight_quant_params, act_quant_params)
        self.quant_to_v = QuantModule(crossattention.to_v,, weight_quant_params, act_quant_params)

        self.quant_to_out = nn.Sequential(
            QuantModule(crossattention.to_out[0],, weight_quant_params, act_quant_params),
            crossattention.to_out[1],
        )

    def forward(self, x, context=None, mask=None):
        h = crossattention.heads

        q = self.quant_to_q(x)
        context = default(context, x)
        k = self.quant_to_k(context)
        v = self.quant_to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * crossattention.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # 是否需要量化
        attn = sim.softmax(dim=-1)

        # 将注意力矩阵用于v, 得到输出
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        out = self.quant_to_out(out)

        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

class QuantFeedForward(BaseQuantBlock):
    def __init__(self, feedforward: FeedForward, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.quant_net = nn.Sequential(
            QuantModule(feedforward.net[0][1], weight_quant_params, act_quant_params),
            feedforward.net[0][2],
            feedforward.net[1],
            QuantModule(feedforward.net[2], weight_quant_params, act_quant_params),
        )   if len(feedforward.net[0]) = 2 else nn.Sequential(
            feedforward.net[0],
            feedforward.net[1],
            QuantModule(feedforward.net[2], weight_quant_params, act_quant_params),
        )

    def forward(self, x):
        return self.quant_net(x)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(self, basictransformerblock: BasicTransformerBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        # 量化模块调量化模块的传入参数 TODO
        self.quant_attn1 = QuantCrossAttention(crossattention: CrossAttention, weight_quant_params, act_quant_params)
        self.quant_ff = QuantFeedForward(feedforward: FeedForward, weight_quant_params, act_quant_params)
        self.quant_attn2 = QuantCrossAttention(crossattention: CrossAttention, weight_quant_params, act_quant_params)

    # 和上文一样：parameters应该用谁的? checkpoint? TODO
    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), basictransformerblock.checkpoint)

    def _forward(self, x, context=None):
        x = self.quant_attn1(basictransformerblock.norm1(x)) + x
        x = self.quant_attn2(basictransformerblock.norm2(x), context=context) + x
        x = self.quant_ff(basictransformerblock.norm3(x)) + x
        return x

class QuantSpatialTransformer(BaseQuantBlock):
    def __init__(self, spatialtransformer: SpatialTransformer, weight_quant_params: dict = {},
                 act_quant_params: dict = {}):
        super().__init__()

        self.quant_proj_in = QuantModule(spatialtransformer.proj_in, weight_quant_params, act_quant_params),

        # TODO: hard


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class _QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, _inv_res: _InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.apply_residual = _inv_res.apply_residual
        self.conv = nn.Sequential(
            QuantModule(_inv_res.layers[0], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[3], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[6], weight_quant_params, act_quant_params, disable_act_quant=True),
        )
        self.conv[0].activation_function = nn.ReLU()
        self.conv[1].activation_function = nn.ReLU()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,
}
