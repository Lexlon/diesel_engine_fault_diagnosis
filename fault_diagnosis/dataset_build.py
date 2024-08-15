import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# 读取10个csv并合并
class build_dataset:
    def __init__(self,file_path = ''):
        self.file_path = file_path

    def _read_data(self,nums = 5):
        '''
        data_set: 读取的数据集
        shape: (features_nums, samples_nums) -> (7200,500*nums)
        '''
        data_set = []
        for num in range(1, nums):
            data = pd.read_csv(self.file_path + f'1800-57%-{num}-Z1.csv')
            data = data.drop(columns=[data.columns[-1]])
            samples = data.iloc[:, 1:].values
            data_set.append(samples)
            # self.samples_labels.append(self.label)
        data_set = np.concatenate(data_set, axis=1)
        return data_set.T
    
    def create_edge_index(self, num_nodes):
        '''
        num_nodes: 节点数量
        节点之间顺序连接
        eg: 5个节点,边索引为[[0,1,2,3,4],[1,2,3,4,5]]
        return: 边索引

        '''
        return torch.stack([torch.arange(num_nodes-1), torch.arange(1, num_nodes)], dim=0)

    
    def construct_graph(self, file_nums, label, num_nodes=5):
        '''
        file_nums: 读取的文件数量
        label: 标签
        num_nodes: 节点数量
        samples: 读取的数据集 (features_nums, samples_nums)
        node_features: 每个节点的特征 node_features = 7200//num_nodes)
        return: graphs (list) -> [Data(x, edge_index, y)]
        Data: x: 节点特征(num_nodes, node_features), edge_index: 边索引[2,num_nodes-1], y: 标签()
        '''
        graphs = []
        edge_index = self.create_edge_index(num_nodes)
        samples = self._read_data(file_nums)
        for i in range(samples.shape[0]):
            sample = samples[i, :]
            nodes = []
            for i in range(num_nodes):
                node_features = sample[i * (7200//num_nodes): (i + 1) * 7200//num_nodes]
                nodes.append(node_features)
        
            # 将节点特征、边索引和标签转为PyTorch Geometric的数据对象
            graph_data = Data(x=torch.tensor(np.array(nodes), dtype=torch.float), 
                            edge_index=edge_index, 
                            y=torch.tensor([label], dtype=torch.long))
            graphs.append(graph_data)
        return graphs


if __name__ == '__main__':
    Build_dataset = build_dataset(file_path='发动机试验数据/高频信号/1800-57%-正常工况/')
    graphs = Build_dataset.construct_graph(file_nums = 5, label = 0)
    for data in graphs:
        print(data.x.shape)
        print(data.edge_index)
        print(data.y)
        break


