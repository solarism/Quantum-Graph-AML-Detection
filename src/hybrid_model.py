import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from src.quantum_circuit import build_quantum_circuit

class HybridQGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits):
        super(HybridQGNN, self).__init__()
        
        # --- 模組二：傳統圖特徵嵌入層 (Encoder) [cite: 52] ---
        # 使用 GraphSAGE 進行歸納式學習與特徵聚合
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        
        # 降維層：將高維特徵壓縮至量子位元數量 (例如 4維) 
        self.compressor = nn.Linear(hidden_dim, n_qubits)
        
        # --- 模組三：量子核心運算層 (Quantum Layer) [cite: 55] ---
        # 建構量子神經網路 (QNN)
        qc, _, _ = build_quantum_circuit(n_qubits)
        qnn = EstimatorQNN(circuit=qc, input_params=qc.parameters[:n_qubits], weight_params=qc.parameters[n_qubits:])
        self.quantum_layer = TorchConnector(qnn)
        
        # 最終分類層 (輸出非法交易機率)
        self.classifier = nn.Linear(1, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        # 1. 傳統 GNN 提取圖特徵
        h = self.sage1(x, edge_index).relu()
        h = self.sage2(h, edge_index).relu()
        
        # 2. 壓縮至適合量子電路的維度 (例如 166 -> 64 -> 4)
        h_compressed = self.compressor(h)
        
        # 3. 進入量子層處理 (VQC)
        # 注意：在 NISQ 時代，這裡通常使用小批次處理以節省模擬資源
        q_out = self.quantum_layer(h_compressed)
        
        # 4. 最終預測
        out = self.classifier(q_out)
        return self.sigmoid(out)