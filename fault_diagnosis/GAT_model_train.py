import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv,  BatchNorm,  global_mean_pool, GCNConv# noqa
import random
from dataset_build import build_dataset
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, feature, out_channel,pooltype):
        super(GAT, self).__init__()

        
        self.GConv1 = GATConv(feature,512)
        self.bn1 = BatchNorm(512)

        self.GConv2 = GATConv(512,512)
        self.bn2 = BatchNorm(512)

        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(256, out_channel))
        self.pooltype = pooltype
        self.pool1, self.pool2 = self.poollayer()


    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x, edge_index, batch = self.poolresult(self.pool1, x, edge_index, batch)
        x1 = global_mean_pool(x, batch)

        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x, edge_index, batch = self.poolresult(self.pool2, x, edge_index, batch)
        x2 = global_mean_pool(x, batch)

        x = x1 + x2
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)

    def poollayer(self):


        if self.pooltype == 'TopKPool':
            self.pool1 = TopKPooling(1024)
            self.pool2 = TopKPooling(1024)
        elif self.pooltype == 'EdgePool':
            self.pool1 = EdgePooling(1024)
            self.pool2 = EdgePooling(1024)
        elif self.pooltype == 'ASAPool':
            self.pool1 = ASAPooling(1024)
            self.pool2 = ASAPooling(1024)
        elif self.pooltype == 'SAGPool':
            self.pool1 = SAGPooling(1024)
            self.pool2 = SAGPooling(1024)
        else:
            print('Such graph pool method is not implemented!!')

        return self.pool1, self.pool2

    def poolresult(self,pool,x,edge_index,batch):

        self.pool = pool

        if self.pooltype == 'TopKPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif self.pooltype == 'EdgePool':
            x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif self.pooltype == 'ASAPool':
            x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif self.pooltype == 'SAGPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        else:
            print('Such graph pool method is not implemented!!')

        return x, edge_index, batch
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # # 第二层图卷积
        x = self.conv2(x, edge_index)
        
        # 池化并生成最终输出
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)
    
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    # correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # pred = out.max(dim=1)[1]
        # correct += pred.eq(data.y).sum().item()
        loss = F.nll_loss(out, data.y.view(-1))  # 交叉熵损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(model,loader,device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    import os
    batch = 128
    num_nodes = 5

    pos_label = 0
    neg1_label = 1
    neg2_label = 2
    neg3_label = 3
    neg4_label = 4
    file_nums = 2
    train_epoch = 50
    current_dir = os.path.dirname(os.path.abspath(__file__))
    normal_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-0.35气门/'
    error1_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-排气门0.5/'
    error2_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-0.4排气门/'
    error3_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-0.2/'
    error4_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-0.5/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normal = build_dataset(normal_file_path)
    pos_graphs = normal.construct_graph(file_nums=file_nums, label=pos_label,num_nodes=num_nodes)
    error1 = build_dataset(error1_file_path)
    neg1_graphs = error1.construct_graph(file_nums=file_nums, label=neg1_label,num_nodes=num_nodes)
    error2 = build_dataset(error2_file_path)
    neg2_graphs = error2.construct_graph(file_nums=file_nums, label=neg2_label,num_nodes=num_nodes)
    error3 = build_dataset(error3_file_path)
    neg3_graphs = error3.construct_graph(file_nums=file_nums, label=neg3_label,num_nodes=num_nodes)
    error4 = build_dataset(error4_file_path)
    neg4_graphs = error4.construct_graph(file_nums=file_nums, label=neg4_label,num_nodes=num_nodes)
    all_graphs = pos_graphs + neg1_graphs + neg2_graphs+neg3_graphs+neg4_graphs
    random.shuffle(all_graphs)
    dataset = all_graphs
    train_dataset = dataset[:int(0.1 * len(dataset))]
    test_dataset = dataset[int(0.5 * len(dataset)):]
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    model = GAT(feature=7200//num_nodes, out_channel=5, pooltype= 'TopKPool').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_train = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(train_epoch):
        loss = train(model, train_loader, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        loss_train.append(loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_Acc: {train_acc:.4f}, Test_Acc: {test_acc:.4f}')

    # plot loss and accuracy
    import matplotlib.pyplot as plt
    train_epoch_list = list(range(train_epoch))
    plt.figure(figsize=(10, 5))
    plt.plot(train_epoch_list, loss_train, color = 'green', label='loss')
    # plt.plot(train_epoch_list, train_acc_list, color = 'blue', label='train accuracy')
    plt.title('Loop')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # save figure
    plt.savefig(current_dir+f'/../result/GAT/GCN_model_3_nodes_{num_nodes}_epoches_{train_epoch}_loss.png')
    # clean figure
    plt.cla()
    plt.plot(train_epoch_list, test_acc_list, color = 'red', label='teat accuracy')
    plt.plot(train_epoch_list, train_acc_list, color = 'blue', label='train accuracy')
    plt.title('Loop')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # save figure
    plt.savefig(current_dir+f'/../result/GAT/GCN_model_3_nodes_{num_nodes}_epoches_{train_epoch}_acc.png')
    # plt.ylabel('loa Amplitude')
  

    




