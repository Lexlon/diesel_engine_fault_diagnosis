import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,  BatchNorm,  global_mean_pool# noqa

from dataprocess.dataset import read_data, create_graph_data
import numpy as np
from torch_geometric.loader import DataLoader
import random
# from GNN.GAT import GAT

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # 聚合每个图的节点输出
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    data_list = ['1500', '1800', '2100', '2300']
    node_dataset = []
    all_graphs = []
    all_labels = []
    for label, data in enumerate(data_list, start=0):
        labeled_data= read_data(data, label)
        all_graphs.extend([node[0] for node in labeled_data])
        all_labels.extend([node[1] for node in labeled_data])
    num_nodes_per_graph = 5  # 每个图有五个节点
    edge_index = np.array(
        [[i, i + 1] for i in range(num_nodes_per_graph - 1)] + [[i + 1, i] for i in range(num_nodes_per_graph - 1)]).T

    graph_datasets = []
    for graph, label in zip(all_graphs, all_labels):
        graph_data = create_graph_data(graph, edge_index, [label])
        graph_datasets.append(graph_data)
    batch_size = 256
    # shuffle graph datasets
    random.shuffle(graph_datasets)
    dataset = graph_datasets
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(in_channels=800, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # model = GAT(in_channels=40, out_channels=4)
    # for data in train_loader:
    #     data = data.to(device)
    #     out = model(data)
    #     print(out)
    #     print(out.shape)
    #     print(data.y.view(-1))
    #
    #     print(out.max(dim=1)[1])
    #     break


    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)


    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.max(dim=1)[1]
            # print(pred)
            # print(data.y)
            correct += pred.eq(data.y).sum().item()
            # print(correct)
        return correct / len(loader.dataset)


    for epoch in range(1, 300):
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} , Test Acc: {test_acc:.4f}')
        # break
