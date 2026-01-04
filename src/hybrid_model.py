import torch
import torch.nn as nn
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
import sys
import os

# 將專案根目錄加入搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantum_circuit import build_quantum_circuit # 保留原本的引用

class HybridQGNN(nn.Module):
    """
    升級版混合量子圖神經網路 (Advanced Hybrid QGNN)
    整合了 SamplerQNN 以獲取量子態機率分佈，並包含前後端傳統處理層。
    """
    def __init__(self, input_dim, hidden_dim, n_qubits=4):
        super(HybridQGNN, self).__init__()
        
        # 1. 傳統預處理層 (Classical Pre-process / Encoder)
        # 將 GraphSAGE 的高維輸出 (input_dim) 壓縮至量子位元數 (n_qubits)
        self.classical_pre_process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh() # 關鍵：將數值限制在 [-1, 1] 區間，對應量子旋轉角度
        )
        
        # 2. 量子層 (Quantum Layer)
        # 定義量子特徵映射 (Feature Map)
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
        
        # 定義變分量子電路 (Ansatz)
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement='linear')
        
        # 組合電路
        qc = feature_map.compose(ansatz)
        
        # 定義 SamplerQNN (基於採樣的神經網路)
        # 輸出維度通常是 2^n_qubits (機率分佈)
        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True # 允許 PyTorch 進行反向傳播
        )
        
        # 轉為 PyTorch 層
        self.quantum_layer = TorchConnector(qnn)
        
        # 3. 傳統後處理層 (Classical Post-process / Classifier)
        # 接收 SamplerQNN 的輸出 (2^4 = 16 維的機率分佈)
        self.classical_post_process = nn.Linear(2**n_qubits, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index=None):
        # 為了相容之前的 GraphSAGE 介面，保留 edge_index 參數，
        # 但在這個 Demo 中我們假設 x 已經是聚合過的特徵
        
        # Step 1: 傳統降維
        x = self.classical_pre_process(x)
        
        # Step 2: 量子計算 (Core)
        x = self.quantum_layer(x)
        
        # Step 3: 最終分類
        x = self.classical_post_process(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    print("初始化升級版 HybridQGNN...")
    model = HybridQGNN(input_dim=64, hidden_dim=32, n_qubits=4)
    print(model)
    print("✅ 模型架構建置成功")