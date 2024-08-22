import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv,  BatchNorm,  global_mean_pool, GCNConv# noqa
import random
from dataset_build import build_dataset

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=4, dropout=0.2)
        self.conv2 = GATConv(16 * 4, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # 聚合每个图的节点输出
        return F.log_softmax(x, dim=1)
    
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
        x = F.dropout(x, p=0.5, training=self.training)
        
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
    batch = 64
    num_nodes = 20
    pos_label = 0
    neg_label = 1
    neg1_label = 2
    file_nums = 2
    train_epoch = 50
    current_dir = os.path.dirname(os.path.abspath(__file__))
    normal_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-0.35气门/'
    error_file_path = current_dir+'/../发动机试验数据/高频信号/1800-57%-0.5/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal = build_dataset(normal_file_path)
    pos_graphs = normal.construct_graph(file_nums=file_nums, label=pos_label,num_nodes=num_nodes)
    error = build_dataset(error_file_path)
    neg_graphs = error.construct_graph(file_nums=file_nums, label=neg_label,num_nodes=num_nodes)
    error1 = build_dataset(error_file_path)
    neg1_graphs = error1.construct_graph(file_nums=file_nums, label=neg1_label,num_nodes=num_nodes)
    all_graphs = pos_graphs + neg_graphs + neg1_graphs
    random.shuffle(all_graphs)
    dataset = all_graphs
    train_dataset = dataset[:int(0.1 * len(dataset))]
    test_dataset = dataset[int(0.5 * len(dataset)):]
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    model = GAT(in_channels=7200//num_nodes, out_channels=3).to(device)
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
    plt.savefig(current_dir+f'/GAT_model_3_nodes_{num_nodes}_epoches_{train_epoch}_loss.png')
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
    plt.savefig(current_dir+f'/GAT_model_3_nodes_{num_nodes}_epoches_{train_epoch}_acc.png')
    # plt.ylabel('loa Amplitude')
  

    




