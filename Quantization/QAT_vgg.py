import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import copy

from tqdm.auto import tqdm
from datasets import *
from quantize_layer import *
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


class QuantizationAwareVGG(nn.Module):
    def __init__(self, original_model, feature_bitwidth=8, weight_bitwidth=8):
        super(QuantizationAwareVGG, self).__init__()
        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

        # 将原始模型的 backbone 转换为支持 QAT 的结构
        quantized_backbone = []
        for layer in original_model.backbone:
            if isinstance(layer, nn.Conv2d):
                quantized_backbone.append(layer)
                quantized_backbone.append(FakeQuantize(feature_bitwidth))
            elif isinstance(layer, nn.ReLU):
                quantized_backbone.append(layer)
            elif isinstance(layer, nn.MaxPool2d):
                quantized_backbone.append(layer)
            elif isinstance(layer, nn.AvgPool2d):
                quantized_backbone.append(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                quantized_backbone.append(layer)
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        self.backbone = nn.Sequential(*quantized_backbone)

        # 对分类器部分也进行类似的处理
        quantized_classifier = [original_model.classifier, FakeQuantize(weight_bitwidth)]
        # for layer in original_model.classifier:
        #     if isinstance(layer, nn.Linear):
        #         quantized_classifier.append(layer)
        #         quantized_classifier.append(FakeQuantize(feature_bitwidth))
        #     elif isinstance(layer, nn.ReLU):
        #         quantized_classifier.append(layer)
        #     else:
        #         raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        self.classifier = nn.Sequential(*quantized_classifier)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def train_qat(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader['train'], desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader['train']):.4f}")

        # 每个 epoch 结束后评估模型性能
        model.eval()
        accuracy = evaluate(model, dataloader['test'])
        print(f"Test Accuracy after epoch {epoch + 1}: {accuracy:.2f}%")
        model.train()

def extract_quantization_params(module):
    """
    Extract quantization parameters (scale and zero_point) from a module.
    :param module: [torch.nn.Module] the module to extract parameters from
    :return:
        A dictionary containing input_scale, input_zero_point, weight_scale, and weight_zero_point
    """
    if hasattr(module, 'weight_fake_quant'):
        # Extract weight scale and zero_point from FakeQuantize module
        weight_scale = module.weight_fake_quant.scale
        weight_zero_point = module.weight_fake_quant.zero_point
    else:
        weight_scale, weight_zero_point = None, None

    if hasattr(module, 'activation_post_process'):
        # Extract input scale and zero_point from activation_post_process
        input_scale = module.activation_post_process.scale
        input_zero_point = module.activation_post_process.zero_point
    else:
        input_scale, input_zero_point = None, None

    return {
        'input_scale': input_scale,
        'input_zero_point': input_zero_point,
        'weight_scale': weight_scale,
        'weight_zero_point': weight_zero_point
    }


def build_custom_quantized_model(qat_model, feature_bitwidth=8, weight_bitwidth=8):
    quantized_backbone = []
    ptr = 0
    while ptr < len(qat_model.backbone):
        layer = qat_model.backbone[ptr]
        if isinstance(layer, nn.Conv2d):
            params = extract_quantization_params(layer)
            quantized_weight, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(
                layer.weight.data, weight_bitwidth
            )
            quantized_bias, bias_scale, bias_zero_point = linear_quantize_bias_per_output_channel(
                layer.bias.data, params['weight_scale'], params['input_scale']
            )
            shifted_quantized_bias = shift_quantized_conv2d_bias(
                quantized_bias, quantized_weight, params['input_zero_point']
            )

            quantized_conv = QuantizedConv2d(
                quantized_weight, shifted_quantized_bias,
                params['input_zero_point'], params['output_zero_point'],
                params['input_scale'], weight_scale, params['output_scale'],
                layer.stride, layer.padding, layer.dilation, layer.groups,
                feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
            )
            quantized_backbone.append(quantized_conv)
            ptr += 1
        elif isinstance(layer, nn.ReLU):
            quantized_backbone.append(layer)
            ptr += 1
        elif isinstance(layer, nn.MaxPool2d):
            quantized_backbone.append(QuantizedMaxPool2d(
                kernel_size=layer.kernel_size, stride=layer.stride
            ))
            ptr += 1
        elif isinstance(layer, nn.AvgPool2d):
            quantized_backbone.append(QuantizedAvgPool2d(
                kernel_size=layer.kernel_size, stride=layer.stride
            ))
            ptr += 1
        else:
            raise NotImplementedError(type(layer))

    # 处理分类器部分
    fc_layer = qat_model.classifier
    params = extract_quantization_params(fc_layer)
    quantized_weight, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(
        fc_layer.weight.data, weight_bitwidth
    )
    quantized_bias, bias_scale, bias_zero_point = linear_quantize_bias_per_output_channel(
        fc_layer.bias.data, params['weight_scale'], params['input_scale']
    )
    shifted_quantized_bias = shift_quantized_linear_bias(
        quantized_bias, quantized_weight, params['input_zero_point']
    )

    quantized_classifier = QuantizedLinear(
        quantized_weight, shifted_quantized_bias,
        params['input_zero_point'], params['output_zero_point'],
        params['input_scale'], weight_scale, params['output_scale'],
        feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
    )

    # 构建最终的量化模型
    quantized_model = copy.deepcopy(qat_model)
    quantized_model.backbone = nn.Sequential(*quantized_backbone)
    quantized_model.classifier = quantized_classifier
    return quantized_model


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

    # Build QAT model
    qat_model = QuantizationAwareVGG(model_fused).cuda()

    # Train QAT model
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_qat(qat_model, dataloader, optimizer, criterion, epochs=3)

    # Build custom quantized model
    custom_quantized_model = build_custom_quantized_model(qat_model)

    # Evaluate quantized model
    quantized_model_accuracy = evaluate(
        custom_quantized_model, dataloader['test'],
        extra_preprocess=[lambda x: extra_preprocess(x, bitwidth=8)]
    )
    print(f"Quantized model accuracy: {quantized_model_accuracy:.2f}%")







