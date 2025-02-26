import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import copy

from quantize_layer import *
from datasets import *
import numpy as np
import random
from vgg import VGG


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def add_range_recoder_hook(model):
    import functools
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}
    def _record_range(self, x, y, module_name):
        x = x[0]
        input_activation[module_name] = x.detach()
        output_activation[module_name] = y.detach()

    all_hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
            all_hooks.append(m.register_forward_hook(
                functools.partial(_record_range, module_name=name)))
    return all_hooks, input_activation, output_activation



if __name__ == '__main__':

    # load pretrained model
    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    model = VGG().cuda()
    print(f"=> loading checkpoint '{checkpoint_url}'")
    model.load_state_dict(checkpoint['state_dict'])
    recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])

    # load dataset
    dataloader = load_cifar10()

    # evaluate the model
    fp32_model_accuracy = evaluate(model, dataloader['test'])
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    print(f"fp32 model has size={fp32_model_size / MiB:.2f} MiB")

    # test the fuse_conv_bn function
    model_fused = test_fuse_conv_bn(model, dataloader)

    """
    We will run the model with some sample data to get the range of each feature map, 
    so that we can get the range of the feature maps and compute their corresponding 
    scaling factors and zero points.
    """

    hooks, input_activation, output_activation = add_range_recoder_hook(model_fused)
    sample_data = iter(dataloader['train']).__next__()[0]
    model_fused(sample_data.cuda())

    # remove hooks
    for h in hooks:
        h.remove()

    """
    3. Finally, let's do model quantization. We will convert the model in the following mapping
    nn.Conv2d: QuantizedConv2d, nn.Linear: QuantizedLinear,
    # the following twos are just wrappers, as current torch modules do not support int8 data format;
    # we will temporarily convert them to fp32 for computation
    nn.MaxPool2d: QuantizedMaxPool2d, nn.AvgPool2d: QuantizedAvgPool2d,
    """
    # we use int8 quantization, which is quite popular
    for bitwidth in [8, 7, 6, 5, 4, 3, 2]:
        feature_bitwidth = weight_bitwidth = bitwidth
        quantized_model = copy.deepcopy(model_fused)
        quantized_backbone = []
        ptr = 0
        while ptr < len(quantized_model.backbone):
            if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and \
                    isinstance(quantized_model.backbone[ptr + 1], nn.ReLU):
                conv = quantized_model.backbone[ptr]
                conv_name = f'backbone.{ptr}'
                relu = quantized_model.backbone[ptr + 1]
                relu_name = f'backbone.{ptr + 1}'

                input_scale, input_zero_point = \
                    get_quantization_scale_and_zero_point(
                        input_activation[conv_name], feature_bitwidth)

                output_scale, output_zero_point = \
                    get_quantization_scale_and_zero_point(
                        output_activation[relu_name], feature_bitwidth)

                quantized_weight, weight_scale, weight_zero_point = \
                    linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
                quantized_bias, bias_scale, bias_zero_point = \
                    linear_quantize_bias_per_output_channel(
                        conv.bias.data, weight_scale, input_scale)
                shifted_quantized_bias = \
                    shift_quantized_conv2d_bias(quantized_bias, quantized_weight,
                                                input_zero_point)

                quantized_conv = QuantizedConv2d(
                    quantized_weight, shifted_quantized_bias,
                    input_zero_point, output_zero_point,
                    input_scale, weight_scale, output_scale,
                    conv.stride, conv.padding, conv.dilation, conv.groups,
                    feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
                )

                quantized_backbone.append(quantized_conv)
                ptr += 2
            elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
                quantized_backbone.append(QuantizedMaxPool2d(
                    kernel_size=quantized_model.backbone[ptr].kernel_size,
                    stride=quantized_model.backbone[ptr].stride
                ))
                ptr += 1
            elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
                quantized_backbone.append(QuantizedAvgPool2d(
                    kernel_size=quantized_model.backbone[ptr].kernel_size,
                    stride=quantized_model.backbone[ptr].stride
                ))
                ptr += 1
            else:
                raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen
        quantized_model.backbone = nn.Sequential(*quantized_backbone)

        # finally, quantized the classifier
        fc_name = 'classifier'
        fc = model.classifier
        input_scale, input_zero_point = \
            get_quantization_scale_and_zero_point(
                input_activation[fc_name], feature_bitwidth)

        output_scale, output_zero_point = \
            get_quantization_scale_and_zero_point(
                output_activation[fc_name], feature_bitwidth)

        quantized_weight, weight_scale, weight_zero_point = \
            linear_quantize_weight_per_channel(fc.weight.data, weight_bitwidth)
        quantized_bias, bias_scale, bias_zero_point = \
            linear_quantize_bias_per_output_channel(
                fc.bias.data, weight_scale, input_scale)
        shifted_quantized_bias = \
            shift_quantized_linear_bias(quantized_bias, quantized_weight,
                                        input_zero_point)

        quantized_model.classifier = QuantizedLinear(
            quantized_weight, shifted_quantized_bias,
            input_zero_point, output_zero_point,
            input_scale, weight_scale, output_scale,
            feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
        )

        # print(quantized_model)

        model_accuracy = evaluate(quantized_model, dataloader['test'],
                                       extra_preprocess=[extra_preprocess])
        print(f"int{bitwidth} model has accuracy={model_accuracy:.2f}%")





