import qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

def build_quantum_circuit(n_qubits, depth=2):
    """
    建構計畫書中提到的量子電路部分
    1. Feature Map: 將古典資料映射到希爾伯特空間 [cite: 83]
    2. Ansatz: 參數化量子線路 (PQC) 用於學習分類邊界 [cite: 87]
    """
    # 1. Quantum Feature Map (ZZ-Feature Map)
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    
    # 2. Variational Circuit (RealAmplitudes or EfficientSU2)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=depth, entanglement='linear')
    
    # 組合電路
    qc = qiskit.QuantumCircuit(n_qubits)
    qc.append(feature_map, range(n_qubits))
    qc.append(ansatz, range(n_qubits))
    
    return qc, feature_map, ansatz