import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union

# Union 联合类型, 表示可以是Union[]中的任何一种类型，最终返回的也是Union[]中的某种类型
#
def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cali_data, awq: bool = False, order: str = 'before', batch_size: int = 256):
    weight_quant, act_quant = act_get_quant_state(order, awq)  # 获取量化的设置选择(before/after/together)
    module.set_quant_state(weight_quant, act_quant)  # 应用量化设置

    # 设置激活值量化器的初始化方式, 为true表示人工初始化(inited manually)
    for t in module.modules():
        # 如果t是QuantModule或BaseQuantBlock, 不进行人工设置初值
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)

    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))  # 设置batch size
    # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda())

    # 清空显存缓冲区
    # 解释：PyTorch是有缓存区的设置的，意思就是一个Tensor就算被释放了，进程也不会把空闲出来的显存还给GPU，而是等待下一个Tensor来填入这一片被释放的空间
    torch.cuda.empty_cache()

    # 设置激活值量化器的初始化方式, 为true表示人工初始化(inited manually)
    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)

'''
设置量化选择: 根据order设置是否量化weight以及是否量化activation
before:   不量化weight, 量化activation
after:    根据awq的值选择是否量化weight, 量化activation
together: 量化weight, 量化activation
:return:  最终的量化选择
'''
def act_get_quant_state(order, awq):
    if order == 'before':
        weight_quant, act_quant = False, True
    elif order == 'after':
        weight_quant, act_quant = awq, True
    elif order == 'together':
        weight_quant, act_quant = True, True
    else:
        raise NotImplementedError
    return weight_quant, act_quant
