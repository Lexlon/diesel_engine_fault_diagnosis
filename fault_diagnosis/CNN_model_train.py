import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset_build import build_dataset
import torch.nn.functional as F
from tqdm import tqdm

# 定义1D CNN分类器
class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (7200 // 2 // 2), 256)  # 更新后的输入大小
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度以适应Conv1d的输入 (batch_size, channels, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten操作
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练模型函数
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # 交叉熵损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试模型函数
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    # 参数设置
    batch_size = 16
    input_size = 1  # 单通道输入
    num_classes = 2  # 分类数
    learning_rate = 0.001
    num_epochs = 50
    file_nums = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('参数设置完成')
    # 数据路径
    normal_file_path = '../发动机试验数据/高频信号/1800-57%-正常工况/'
    error_file_path = '../发动机试验数据/高频信号/1800-57%-断缸/'
    
    # 构建数据集
    normal = build_dataset(normal_file_path)
    pos_data = normal._read_data(nums=file_nums)
    error = build_dataset(error_file_path)
    neg_data = error._read_data(nums=file_nums)
    
    X = np.concatenate((pos_data, neg_data), axis=0)
    y = np.concatenate((np.zeros(pos_data.shape[0]), np.ones(neg_data.shape[0])), axis=0)  # 0表示正常，1表示断缸
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42, shuffle=True)

    # 将数据转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # 增加一维以适应Conv1d输入
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 使用 DataLoader 构建数据加载器
    train_dataset = TensorDataset(X_train[:30], y_train[:30])
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('数据集构建完成')
    # 初始化模型、优化器
    model = CNN1DClassifier(input_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('模型构建完成')
    # 训练模型
    loss_train = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        train_accuracy = test(model, train_loader, device)
        test_accuracy = test(model, test_loader, device)
        loss_train.append(train_loss)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    import matplotlib.pyplot as plt

    train_epoch_list = list(range(num_epochs))
    plt.figure(figsize=(10, 5))
    plt.plot(train_epoch_list, loss_train, color='green', label='loss')
    # plt.plot(train_epoch_list, train_acc_list, color = 'blue', label='train accuracy')
    plt.title('Loop')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # save figure
    plt.savefig(f'CNN_model_epoches_{num_epochs}_loss.png')
    # clean figure
    plt.cla()
    plt.plot(train_epoch_list, test_acc_list, color='red', label='teat accuracy')
    plt.plot(train_epoch_list, train_acc_list, color='blue', label='train accuracy')
    plt.title('Loop')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # save figure
    plt.savefig(f'CNN_model_epoches_{num_epochs}_acc.png')