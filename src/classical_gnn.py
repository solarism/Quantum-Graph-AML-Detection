import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """
    傳統 GraphSAGE 模型 (Baseline Model)
    用於與 Hybrid Quantum Model 進行效能比較 [Source 123]
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # 定義兩層 GraphSAGE 卷積層 [Source 53]
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一層聚合 + ReLU 激活
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        
        # 第二層聚合
        h = self.conv2(h, edge_index)
        h = h.relu()
        
        # 分類層
        out = self.classifier(h)
        return torch.sigmoid(out)