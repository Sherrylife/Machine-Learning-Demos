import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 超参数
batch_size = 100
image_size = 28
num_latents = 32
classes_per_latent = 10
epochs = 30
tau_initial = 1.0
tau_min = 0.01
tau_decay_rate = 0.999

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='../VQVAE/data/mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../VQVAE/data/mnist', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Gumbel Softmax重参数化
def gumbel_softmax(logits, tau, hard=False):
    epsilon = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(epsilon + 1e-8) + 1e-8)
    y = logits + gumbel_noise
    y_soft = F.softmax(y / tau, dim=-1)

    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


# 编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 32)
        self.fc_logits = nn.Linear(32, num_latents * classes_per_latent)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        logits = logits.view(-1, num_latents, classes_per_latent)
        return logits


# 解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(num_latents * classes_per_latent, 32)
        self.fc2 = nn.Linear(32, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = z.view(z.size(0), -1)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


# 模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, tau, hard=False):
        logits = self.encoder(x)
        z_sample = gumbel_softmax(logits, tau, hard=hard)
        x_recon = self.decoder(z_sample)
        return x_recon, logits


# 损失函数
def loss_function(x, x_recon, logits):
    # 重构损失
    xent_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL散度（假设先验分布为均匀分布）
    p = F.softmax(logits, dim=-1)
    p = torch.clamp(p, min=1e-8, max=1 - 1e-8)
    kl_loss = torch.sum(p * torch.log(p), dim=[1, 2]).mean()

    return xent_loss + kl_loss


# 初始化模型、优化器和tau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
tau = tau_initial


# 训练过程
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        x_recon, logits = model(data, tau)
        loss = loss_function(data, x_recon, logits)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 更新tau
    tau = max(tau_min, tau * tau_decay_rate)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}, Tau: {tau:.5f}')


# 测试生成图像
model.eval()
with torch.no_grad():
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    for i in range(n):
        for j in range(n):
            z_sample = torch.zeros((1, num_latents, classes_per_latent), device=device)
            for iz in range(num_latents):
                jz = np.random.choice(classes_per_latent)
                z_sample[0, iz, jz] = 1
            x_decoded = model.decoder(z_sample).cpu().numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size:(i + 1) * digit_size,
                   j * digit_size:(j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()