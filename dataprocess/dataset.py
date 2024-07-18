import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def read_data(speed,label = 1):
    labeled_data = []
    data = pd.read_csv('data/minmax_'+speed+'.csv')
    window_lenth = 800
    # 划分样本
    num_windows = len(data)//window_lenth
    for i in tqdm(range(num_windows)):
        start_idx = i*window_lenth
        end_idx = (i+1)*window_lenth

        window_data = data.iloc[start_idx:end_idx].copy()
        if not window_data.empty:
            # 为每个样本打上标签
            window_features = window_data[['AI1-02 [m/s²]', 'AI1-05 [m/s²]',
                                                    'AI1-08 [m/s²]', 'AI1-11 [m/s²]',
                                                    'AI1-14 [m/s²]']].T.values
            labeled_data.append((window_features, label))
    # labeled_dataset = pd.concat(labeled_data, ignore_index=True)
    return labeled_data

def create_graph_data(node_features, edge_index, labels):
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == '__main__':

    data_list = ['1500', '1800', '2100', '2300']
    node_dataset = []
    all_graphs = []
    all_labels = []
    for label, data in enumerate(data_list, start=1):
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

    print(f"Total graphs created: {len(graph_datasets)}")
    first_graph = graph_datasets[-1]
    print("First graph details:")
    print(f"x (node features):\n{first_graph.x.shape}")
    print(f"edge_index (edges):\n{first_graph.edge_index}")
    print(f"y (label):\n{first_graph.y}")