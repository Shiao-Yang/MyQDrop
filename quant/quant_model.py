import torch.nn as nn
from .quant_block import specials, BaseQuantBlock
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from .fold_bn import search_fold_and_remove_bn

'''
量化版的Model, 继承了nn.Module类, 用来将普通的conv2d和Linear层替换为量化版的conv2d和Linear层
'''
class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    # 重构模块, 即用量化后的module替换原module, 权重和激活值也会被替换
    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            # 如果是specials中的模块, 则用量化后的模块替换原模块
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))  # specials[]后的括号是初始化类的参数

            # 如果是Conv2d或Linear, 用QuantModule替换原module, 并将prev_quantmodule设置为替换后的QuantModule.
            # QuantModule: 可以执行量化版的卷积或普通卷积
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            # 如果是ReLU或ReLU6, 且prev_quantmodule是量化版的Conv2d或Linear, 设置prev_quantmodule.activation_function为relu/relu6, 去掉原relu
            # TODO: figure out the meaning
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            # 如果以上情况都不是, 对当前模块执行quant_module_refactor, 即递归量化
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        Recursively set whether to quantize weight and whether to quantize activation in children modules
        :param weight_quant: whether to quantize weight
        :param act_quant: whether to quantize activation
        """
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):  # 如果是QuantModule或者BaseQuantBlock, 设置量化选择
                m.set_quant_state(weight_quant, act_quant)

    # 前向传播
    def forward(self, input):
        return self.model(input)

    # 把第一层和最后一层的weight设置为8bit, 把倒数第二层的activation设置为8bit
    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        'the image input has been in 0~255, set the last layer\'s input to 8-bit'
        a_list[-2].bitwidth_refactor(8)

    # 输出不进行量化
    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
