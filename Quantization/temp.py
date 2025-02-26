import os
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

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


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max


def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    """
    Linear quantization for single fp_tensor.
    :param fp_tensor: [torch.(cuda.)FloatTensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :param scale: [torch.(cuda.)FloatTensor] scaling factor
    :param zero_point: [torch.(cuda.)IntTensor] the desired centroid of tensor values
    :return: [torch.(cuda.)FloatTensor] quantized tensor whose values are integers
    """
    assert fp_tensor.dtype == torch.float
    assert isinstance(scale, float) or (scale.dtype == torch.float and scale.dim() == fp_tensor.dim())
    assert isinstance(zero_point, int) or (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim())

    # Step 1: Scale the fp_tensor
    scaled_tensor = fp_tensor / scale
    # Step 2: Round the floating value to integer value
    rounded_tensor = torch.round(scaled_tensor)
    rounded_tensor = rounded_tensor.to(dtype)

    # Step 3: Shift the rounded_tensor to make zero_point 0
    shifted_tensor = rounded_tensor + zero_point

    # Step 4: Clamp the shifted_tensor to lie in bitwidth-bit range
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    return quantized_tensor


class FakeQuantize(torch.nn.Module):
    def __init__(self, bitwidth=8):
        super(FakeQuantize, self).__init__()
        self.bitwidth = bitwidth
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0))

    def forward(self, x):
        if self.training:
            # In training, dynamically compute scale and zero_point
            scale, zero_point = get_quantization_scale_and_zero_point(x, self.bitwidth)
            self.scale.fill_(scale)
            self.zero_point.fill_(zero_point)

        # Simulate quantization using linear_quantize
        quantized_x = linear_quantize(x, self.bitwidth, self.scale.item(), self.zero_point.item())
        return quantized_x.float()  # Return float for continued training


class QuantizationAwareVGG(nn.Module):
    def __init__(self, original_model, feature_bitwidth=8, weight_bitwidth=8):
        super(QuantizationAwareVGG, self).__init__()
        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

        # Convert backbone to support QAT
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
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
        self.backbone = nn.Sequential(*quantized_backbone)

        # Convert classifier to support QAT
        quantized_classifier = []
        for layer in original_model.classifier:
            if isinstance(layer, nn.Linear):
                quantized_classifier.append(layer)
                quantized_classifier.append(FakeQuantize(feature_bitwidth))
            elif isinstance(layer, nn.ReLU):
                quantized_classifier.append(layer)
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
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
        for inputs, targets in tqdm(dataloader['train'], desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader['train']):.4f}")

        # Evaluate model performance after each epoch
        model.eval()
        accuracy = evaluate(model, dataloader['test'])
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        model.train()


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

    # Handle classifier part
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

    # Build the final quantized model
    quantized_model = copy.deepcopy(qat_model)
    quantized_model.backbone = nn.Sequential(*quantized_backbone)
    quantized_model.classifier = quantized_classifier
    return quantized_model


def evaluate(model, dataloader, extra_preprocess=None):
    model.eval()
    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        inputs = inputs.cuda()
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)
        targets = targets.cuda()

        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)

        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


def load_cifar10():
    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="../data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=512,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )
    return dataloader


if __name__ == '__main__':
    # Load pretrained model
    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    model = VGG().cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # Load dataset
    dataloader = load_cifar10()

    # Evaluate the FP32 model
    fp32_model_accuracy = evaluate(model, dataloader['test'])
    print(f"FP32 model has accuracy={fp32_model_accuracy:.2f}%")

    # Fuse Conv-BN layers
    model_fused = test_fuse_conv_bn(model)

    # Add hooks to record activation ranges
    input_activation = {}
    output_activation = {}
    hooks = add_range_recoder_hook(model_fused)
    sample_data = iter(dataloader['train']).__next__()[0]
    model_fused(sample_data.cuda())
    for h in hooks:
        h.remove()

    # Train QAT model
    qat_model = QuantizationAwareVGG(model_fused).cuda()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_qat(qat_model, dataloader, optimizer, criterion, epochs=5)

    # Build custom quantized model
    custom_quantized_model = build_custom_quantized_model(qat_model)

    # Evaluate quantized model
    quantized_model_accuracy = evaluate(
        custom_quantized_model, dataloader['test'],
        extra_preprocess=[lambda x: extra_preprocess(x, bitwidth=8)]
    )
    print(f"Quantized model accuracy: {quantized_model_accuracy:.2f}%")