import torch
import torch.nn as nn

# 定义 FakeQuantize 类
class FakeQuantize(torch.nn.Module):
    def __init__(self, bitwidth=8):
        super(FakeQuantize, self).__init__()
        self.bitwidth = bitwidth
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0))

    def forward(self, x):
        if self.training:
            # 动态计算 scale 和 zero_point
            scale, zero_point = get_quantization_scale_and_zero_point(x, self.bitwidth)
            self.scale.fill_(scale)
            self.zero_point.fill_(zero_point)

        # 使用 Straight-Through Estimator (STE) 进行量化
        quantized_x = self._fake_quantize(x, self.bitwidth, self.scale.item(), self.zero_point.item())
        return quantized_x

    def _fake_quantize(self, x, bitwidth, scale, zero_point):
        """
        Simulate quantization with Straight-Through Estimator (STE).
        :param x: [torch.Tensor] input tensor
        :param bitwidth: [int] quantization bit width
        :param scale: [float] scaling factor
        :param zero_point: [int] zero point
        :return: [torch.Tensor] fake-quantized tensor
        """
        # Step 1: Scale the input
        scaled_x = x / scale

        # Step 2: Round the floating value to integer value
        rounded_x = torch.round(scaled_x)

        # Step 3: Shift the rounded value to make zero_point 0
        shifted_x = rounded_x + zero_point

        # Step 4: Clamp the shifted value to lie in bitwidth-bit range
        quantized_min, quantized_max = get_quantized_range(bitwidth)
        clamped_x = shifted_x.clamp(quantized_min, quantized_max)

        # Step 5: Dequantize back to floating-point for continued training
        dequantized_x = (clamped_x - zero_point) * scale

        # Use STE to bypass gradient computation for quantization
        return x + (dequantized_x - x).detach()


# 定义辅助函数
def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max


def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = round(quantized_min - fp_min / scale)

    # Clip zero_point to fall in [quantized_min, quantized_max]
    zero_point = max(quantized_min, min(quantized_max, zero_point))
    return scale, int(zero_point)


# 测试 FakeQuantize 的反向传播
if __name__ == "__main__":
    # 创建一个简单的模型
    torch.manual_seed(0)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(2, 3, bias=False)
            self.fake_quant = FakeQuantize(bitwidth=8)

        def forward(self, x):
            x = self.fc(x)
            x = self.fake_quant(x)
            return x

    # 初始化模型和损失函数
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 输入数据和目标值
    inputs = torch.randn(1, 2, requires_grad=True)  # 需要梯度的输入
    # targets = torch.randn(1, 3)
    targets = torch.tensor([1,2,3])

    # 前向传播
    print("model", list(model.parameters()))
    print("input", inputs)
    outputs = model(inputs)
    print("output", outputs)
    loss = torch.mean(outputs)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 检查梯度是否存在
    print("Gradients of the linear layer's weights:")
    print(model.fc.weight.grad)

    print("Gradients of the input tensor:")
    print(inputs.grad)