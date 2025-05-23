import torch
import torch.nn as nn
import copy
from linear_quantization import *
from datasets import *

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, method, bitwidth):
        if method == "per_channel":
            weight_q, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(x, bitwidth)
            weight_q_dq = linear_dequantize_weight_per_channel(weight_q, weight_scale, weight_zero_point)
            return weight_q_dq
        elif method == "all":
            scale, zero_point = get_quantization_scale_and_zero_point(x, bitwidth)
            x_q = linear_quantize(x, bitwidth, scale, zero_point)
            x_q_dq = linear_dequantize(x_q, bitwidth, scale, zero_point)
            return x_q_dq
        else:
            raise NotImplementedError("Error STE quantization method")

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

# class FakeQuantize(torch.nn.Module):
#     def __init__(self, bitwidth=8):
#         super(FakeQuantize, self).__init__()
#         self.bitwidth = bitwidth
#         self.register_buffer('scale', torch.tensor(1.0))
#         self.register_buffer('zero_point', torch.tensor(0))
#
#     def forward(self, x):
#         if self.training:
#             # 动态计算 scale 和 zero_point
#             scale, zero_point = get_quantization_scale_and_zero_point(x, self.bitwidth)
#             self.scale.fill_(scale)
#             self.zero_point.fill_(zero_point)
#
#         # 使用 Straight-Through Estimator (STE) 进行量化
#         quantized_x = self._fake_quantize(x, self.bitwidth, self.scale.item(), self.zero_point.item())
#         return quantized_x
#
#     def _fake_quantize(self, x, bitwidth, scale, zero_point):
#         """
#         Simulate quantization with Straight-Through Estimator (STE).
#         :param x: [torch.Tensor] input tensor
#         :param bitwidth: [int] quantization bit width
#         :param scale: [float] scaling factor
#         :param zero_point: [int] zero point
#         :return: [torch.Tensor] fake-quantized tensor
#         """
#         # Step 1: Scale the input
#         scaled_x = x / scale
#
#         # Step 2: Round the floating value to integer value
#         rounded_x = torch.round(scaled_x)
#
#         # Step 3: Shift the rounded value to make zero_point 0
#         shifted_x = rounded_x + zero_point
#
#         # Step 4: Clamp the shifted value to lie in bitwidth-bit range
#         quantized_min, quantized_max = get_quantized_range(bitwidth)
#         clamped_x = shifted_x.clamp(quantized_min, quantized_max)
#
#         # Step 5: Dequantize back to floating-point for continued training
#         dequantized_x = (clamped_x - zero_point) * scale
#
#         # Use STE to bypass gradient computation for quantization
#         return x + (dequantized_x - x).detach()


class QuantizedConv2d(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 stride, padding, dilation, groups,
                 feature_bitwidth=8, weight_bitwidth=8, qat_training=False):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth
        self.qat_training = qat_training



    def forward(self, x):
        if self.qat_training:
            # qat training
            weight_q, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(self.weight,self.weight_bitwidth)
            weight_q_dq = linear_dequantize_weight_per_channel(weight_q, weight_scale, weight_zero_point)
            self.weight = weight_q_dq
            return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        else:  # integer-only inference
            return quantized_conv2d(
                x, self.weight, self.bias,
                self.feature_bitwidth, self.weight_bitwidth,
                self.input_zero_point, self.output_zero_point,
                self.input_scale, self.weight_scale, self.output_scale,
                self.stride, self.padding, self.dilation, self.groups
                )

class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8, qat_training=False):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth
        self.qat_training = qat_training

    def forward(self, x):
        if self.qat_training:
            # qat training
            weight_q, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(self.weight,self.weight_bitwidth)
            weight_q_dq = linear_dequantize_weight_per_channel(weight_q, weight_scale, weight_zero_point)
            self.weight = weight_q_dq
            return torch.nn.functional.linear(x, self.weight, self.bias)


        return quantized_linear(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
            )

class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AvgPool
        return super().forward(x.float()).to(torch.int8)


def extra_preprocess(x, bitwidth=8):
    # hint: you need to convert the original fp32 input of range (0, 1)
    #  into int8 format of range (-128, 127)
    ############### YOUR CODE STARTS HERE ###############
    x_scale, x_zero_point = get_quantization_scale_and_zero_point(x, bitwidth)
    output = linear_quantize(x, bitwidth, x_scale, x_zero_point)
    qmin, qmax = get_quantized_range(bitwidth)
    return output.clamp(qmin, qmax).to(torch.int8)
    # return (x * 255.0 - 128).clamp(-128, 127).to(torch.int8)
    ############### YOUR CODE ENDS HERE #################

def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


def get_quantization_scale_for_weight(weight, bitwidth):
    """
    get quantization scale for single tensor of weight
    :param weight: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [integer] quantization bit width
    :return:
        [floating scalar] scale
    """
    # we just assume values in weight are symmetric
    # we also always make zero_point 0 for weight
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max

def linear_quantize_weight_per_channel(tensor, bitwidth):
    """
    linear quantization for weight tensor
        using different scales and zero_points for different output channels
    :param tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [torch.(cuda.)Tensor] scale tensor
        [int] zero point (which is always 0)
    """
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)
    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0


def linear_dequantize_weight_per_channel(quantized_tensor, scale, zero_point=0, bitwidth=None):
    """
    Linear dequantization for weight tensor quantized per channel.

    :param quantized_tensor: [torch.(cuda.)Tensor] quantized weight tensor (integer values)
    :param scale: [torch.(cuda.)Tensor] scaling factor for each output channel
    :param zero_point: [int] zero point used during quantization (default is 0)
    :param bitwidth: [int] quantization bit width (optional, for validation)
    :return:
        [torch.(cuda.)Tensor] dequantized floating-point tensor
    """
    # Ensure the input tensor is of integer type
    assert quantized_tensor.dtype in [torch.int8, torch.int16, torch.int32], \
        "quantized_tensor must be of integer type"
    # Validate scale dimensions
    dim_output_channels = 0
    num_output_channels = quantized_tensor.shape[dim_output_channels]
    assert scale.shape[0] == num_output_channels, "Scale tensor must match the number of output channels"
    shifted_tensor = quantized_tensor - zero_point
    fp_tensor = shifted_tensor.to(torch.float) * scale

    return fp_tensor



def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):
    """
    linear quantization for single bias tensor
        quantized_bias = fp_bias / bias_scale
    :param bias: [torch.FloatTensor] bias weight to be quantized
    :param weight_scale: [float or torch.FloatTensor] weight scale tensor
    :param input_scale: [float] input scale
    :return:
        [torch.IntTensor] quantized bias tensor
    """
    assert(bias.dim() == 1)
    assert(bias.dtype == torch.float)
    assert(isinstance(input_scale, float))
    if isinstance(weight_scale, torch.Tensor):
        assert(weight_scale.dtype == torch.float)
        weight_scale = weight_scale.view(-1)
        assert(bias.numel() == weight_scale.numel())

    ############### YOUR CODE STARTS HERE ###############
    # hint: one line of code
    bias_scale = weight_scale * input_scale
    ############### YOUR CODE ENDS HERE #################

    quantized_bias = linear_quantize(bias, 32, bias_scale,
                                     zero_point=0, dtype=torch.int32)
    return quantized_bias, bias_scale, 0


def linear_dequantize_bias_per_output_channel(quantized_bias, bias_scale, zero_point=0):
    """
    Linear dequantization for single bias tensor.

    :param quantized_bias: [torch.IntTensor] quantized bias tensor (integer values)
    :param bias_scale: [float or torch.FloatTensor] bias scale tensor
    :param zero_point: [int] zero point used during quantization (default is 0)
    :return:
        [torch.FloatTensor] dequantized floating-point bias tensor
    """
    # Ensure the input tensor is of integer type
    assert quantized_bias.dtype == torch.int32, \
        "quantized_bias must be of type torch.int32"
    # Validate bias_scale type and shape
    if isinstance(bias_scale, torch.Tensor):
        assert bias_scale.dtype == torch.float, "bias_scale must be of type torch.float"
        bias_scale = bias_scale.view(-1)  # Ensure it's a 1D tensor
        assert quantized_bias.numel() == bias_scale.numel(), \
            "bias_scale must match the number of elements in quantized_bias"
    # Step 1: Shift the quantized_bias back by subtracting zero_point
    shifted_bias = quantized_bias - zero_point
    # Step 2: Scale the shifted_bias using the bias_scale
    fp_bias = shifted_bias.to(torch.float) * bias_scale

    return fp_bias




def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    shift quantized bias to incorporate input_zero_point for nn.Linear
        shifted_quantized_bias = quantized_bias - Linear(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert(quantized_bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point

def shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    shift quantized bias to incorporate input_zero_point for nn.Conv2d
        shifted_quantized_bias = quantized_bias - Conv(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert(quantized_bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    return quantized_bias - quantized_weight.sum((1,2,3)).to(torch.int32) * input_zero_point


def quantized_linear(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale):
    """
    quantized fully-connected layer
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.CharIntTensor] quantized output feature (torch.int8)
    """
    assert(input.dtype == torch.int8)
    assert(weight.dtype == input.dtype)
    assert(bias is None or bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    assert(isinstance(output_zero_point, int))
    assert(isinstance(input_scale, float))
    assert(isinstance(output_scale, float))
    assert(weight_scale.dtype == torch.float)

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    if 'cpu' in input.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.linear(input.to(torch.int32), weight.to(torch.int32), bias)
    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

    ############### YOUR CODE STARTS HERE ###############
    # Step 2: scale the output
    #         hint: 1. scales are floating numbers, we need to convert output to float as well
    #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc]
    output = output * (input_scale * weight_scale.reshape(weight.shape[0]) / output_scale)
    # output = output.to(float) * input_scale * weight_scale.view(1, -1) / output_scale
    # Step 3: shift output by output_zero_point
    #         hint: one line of code
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


def quantized_conv2d(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale,
                     stride, padding, dilation, groups):
    """
    quantized 2d convolution
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.(cuda.)CharTensor] quantized output feature
    """
    assert(len(padding) == 4)
    assert(input.dtype == torch.int8)
    assert(weight.dtype == input.dtype)
    assert(bias is None or bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    assert(isinstance(output_zero_point, int))
    assert(isinstance(input_scale, float))
    assert(isinstance(output_scale, float))
    assert(weight_scale.dtype == torch.float)

    # Step 1: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
    input = torch.nn.functional.pad(input, padding, 'constant', input_zero_point)
    if 'cpu' in input.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.conv2d(input.to(torch.int32), weight.to(torch.int32), None, stride, 0, dilation, groups)
    else:
        # current version pytorch does not yet support integer-based conv2d() on GPUs
        output = torch.nn.functional.conv2d(input.float(), weight.float(), None, stride, 0, dilation, groups)
        output = output.round().to(torch.int32)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    ############### YOUR CODE STARTS HERE ###############
    # hint: this code block should be the very similar to quantized_linear()

    # Step 2: scale the output
    #         hint: 1. scales are floating numbers, we need to convert output to float as well
    #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc, height, width]
    output = output.float() * (input_scale * weight_scale.reshape(weight.shape[0], 1, 1) / output_scale)
    # output = output.to(float) * input_scale * weight_scale.view(1, -1, 1, 1) / output_scale

    # Step 3: shift output by output_zero_point
    #         hint: one line of code
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


def fuse_conv_bn(conv, bn):
    # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

    return conv


def test_fuse_conv_bn(model, dataloader):
    """
    fuse the conv and bn layer for quantization,
    test the accuracy of the fused model and
    return the fused model
    :param model: the original model
    :return: fused model
    """
    print('Before conv-bn fusion: backbone length', len(model.backbone))
    #  fuse the batchnorm into conv layers
    model_fused = copy.deepcopy(model)
    fused_backbone = []
    ptr = 0
    while ptr < len(model_fused.backbone):
        if isinstance(model_fused.backbone[ptr], nn.Conv2d) and \
            isinstance(model_fused.backbone[ptr + 1], nn.BatchNorm2d):
            fused_backbone.append(fuse_conv_bn(
                model_fused.backbone[ptr], model_fused.backbone[ptr+ 1]))
            ptr += 2
        else:
            fused_backbone.append(model_fused.backbone[ptr])
            ptr += 1
    model_fused.backbone = nn.Sequential(*fused_backbone)

    print('After conv-bn fusion: backbone length', len(model_fused.backbone))
    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, nn.BatchNorm2d)

    #  the accuracy will remain the same after fusion
    fused_acc = evaluate(model_fused, dataloader['test'])
    print(f'Accuracy of the fused model={fused_acc:.2f}%')
    return model_fused
