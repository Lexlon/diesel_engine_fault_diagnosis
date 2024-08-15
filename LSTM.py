import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取和合并数据
def load_data():
    positive_samples, negative_samples = [], []
    
    for num in range(1, 2):
        pos_file_path = f'发动机试验数据/高频信号/1800-57%-正常工况/1800-57%-{num}-X11.csv'
        neg_file_path = f'发动机试验数据/高频信号/1800-57%-断缸/1800-57%-{num}-X11.csv'
        
        pos_data = pd.read_csv(pos_file_path).iloc[:, 1:-1].values
        neg_data = pd.read_csv(neg_file_path).iloc[:, 1:-1].values
        
        positive_samples.append(pos_data)
        negative_samples.append(neg_data)
    
    positive_samples = np.concatenate(positive_samples, axis=1).T  # 转置后每行为一个样本
    negative_samples = np.concatenate(negative_samples, axis=1).T  # 转置后每行为一个样本
    
    return positive_samples[:200], negative_samples[:200]

positive_samples, negative_samples = load_data()
print('数据加载完成')
# 创建标签
positive_labels = np.zeros(positive_samples.shape[0])  # 标签0表示正常工况
negative_labels = np.ones(negative_samples.shape[0])   # 标签1表示断缸

# 合并样本和标签
X = np.concatenate((positive_samples, negative_samples), axis=0)
y = np.concatenate((positive_labels, negative_labels), axis=0)

# 数据标准化
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 调整形状为 LSTM 输入格式
X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# LSTM 模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out

# 模型参数
input_size = 1
hidden_size = 128  # 减小 hidden_size 以减少内存占用
num_layers = 1     # 减少层数
num_classes = 2
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print('模型构建完成')
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    print('开始训练')
    for i in range(0, X_train.size(0), batch_size):
        X_batch = X_train[i:i+batch_size].to(device)
        y_batch = y_train[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{X_train.size(0)}], Loss: {loss.item():.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = X_test.size(0)
        for i in range(0, total, batch_size):
            X_batch = X_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = X_test.size(0)
    for i in range(0, total, batch_size):
        X_batch = X_test[i:i+batch_size].to(device)
        y_batch = y_test[i:i+batch_size].to(device)
        
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')