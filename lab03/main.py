import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_set, val_set = random_split(full_train_set, [55000, 5000])

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(B, 16, 49)
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def measure_latency(model, input_tensor, runs, device):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    for _ in range(10):
        _ = model(input_tensor)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(input_tensor)
            latencies.append((time.time() - start) * 1000)

    latencies.sort()
    return latencies[len(latencies) // 2]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {'MLP': MLP(), 'CNN': CNN(), 'Attention': AttentionNet()}
metrics = {}

dummy_input = torch.randn(1, 1, 28, 28)
dummy_batch = torch.randn(128, 1, 28, 28)

for name, model in models.items():
    print(f"\n--- Model: {name} ---")
    summary(model, input_size=(1, 1, 28, 28))

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2

    lat_1 = measure_latency(model, dummy_input, 100, device)
    lat_128 = measure_latency(model, dummy_batch, 20, device)

    metrics[name] = {'params': params, 'flops': flops, 'lat_1': lat_1, 'lat_128': lat_128}
    print(f"THOP FLOPs: {flops:,.0f} | Params: {params:,.0f}")
    print(f"Latency (Batch 1): {lat_1:.2f} ms | Latency (Batch 128): {lat_128:.2f} ms")


def train_model(model, name):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nTraining {name}...")
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    metrics[name]['accuracy'] = acc
    print(f"{name} Test Accuracy: {acc:.2f}%")


for name, model in models.items():
    train_model(model, name)

names = list(metrics.keys())
params = [metrics[n]['params'] for n in names]
latencies = [metrics[n]['lat_128'] for n in names]
accuracies = [metrics[n]['accuracy'] for n in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(params, accuracies, c=['red', 'blue', 'green'], s=100)
for i, name in enumerate(names):
    ax1.annotate(name, (params[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
ax1.set_xlabel('Parameters')
ax1.set_ylabel('Test Accuracy (%)')
ax1.set_title('Accuracy vs Parameters')

ax2.scatter(latencies, accuracies, c=['red', 'blue', 'green'], s=100)
for i, name in enumerate(names):
    ax2.annotate(name, (latencies[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
ax2.set_xlabel('Latency (Batch 128) [ms]')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Accuracy vs Latency')

plt.tight_layout()
plt.savefig('architecture_tradeoffs.png')
plt.show()