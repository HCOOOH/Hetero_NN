import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

file_path = 'guest_b.csv'  # 请替换成你的 CSV 文件路径
data = pd.read_csv(file_path)
features = data[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
labels = data['y']

# 转换为张量
train_features = torch.tensor(features.values, dtype=torch.float32)
train_labels = torch.tensor(labels.values, dtype=torch.long)

file_path = 'guest_b_test.csv'  # 请替换成你的 CSV 文件路径
data = pd.read_csv(file_path)
features = data[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
labels = data['y']

# 转换为张量
test_features = torch.tensor(features.values, dtype=torch.float32)
test_labels = torch.tensor(labels.values, dtype=torch.long)

# 定义模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = F.softmax(self.fc2(out))
        return out

# 初始化模型
input_size = 9  # 输入特征数量
hidden_size = 16  # 隐藏层神经元数量
output_size = 4  # 输出类别数量

model = SimpleMLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将数据加载到 DataLoader
batch_size = 32
train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
num_epochs = 100
for epoch in range(1,num_epochs+1):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:  # 每 10 个批次输出一次
            print(f"Epoch [{epoch}/{num_epochs}], "
                  f"Step [{i + 1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

    if epoch % 25 == 0:
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            pred = torch.argmax(outputs,dim=1)
            correct = (pred == labels).sum().item()
            total = outputs.size(0)
            accuracy = correct / total

            print(f"Accuracy on test set: {accuracy * 100:.2f}%")


print("Finished Training")