import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset_build import build_dataset

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out
    
if __name__ == '__main__':
    batch = 16
    pos_label = 0
    neg_label = 1
    file_nums = 2
    train_epoch = 50
    normal_file_path = '发动机试验数据/高频信号/1800-57%-正常工况/'
    error_file_path = '发动机试验数据/高频信号/1800-57%-断缸/'
    normal = build_dataset(normal_file_path)
    pos_data = normal._read_data(nums=file_nums)
    error = build_dataset(error_file_path)
    neg_data = error._read_data(nums=file_nums)
    # print(pos_data.shape, neg_data.shape)
    X = np.concatenate((pos_data, neg_data), axis=0)
    y = np.concatenate((np.zeros(pos_data.shape[0]), np.ones(neg_data.shape[0])), axis=0)  # 0表示正常，1表示断缸
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

    # 将数据转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # 增加一维以适应LSTM输入
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 使用 DataLoader 构建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
