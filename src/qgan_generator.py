from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.optimizers import SPSA
# 注意：qiskit_algorithms 是新版套件架構
# 這裡封裝成一個類別以便未來擴充

class QuantumDataGenerator:
    """
    基於 EfficientSU2 與 SPSA 最佳化函數的 QGAN 生成器配置。
    用於生成合成洗錢樣本以解決資料不平衡 (Mode Collapse) 問題。
    """
    def __init__(self, num_qubits=4, reps=3):
        self.num_qubits = num_qubits
        self.reps = reps
        self.generator_circuit = self._build_generator()
        self.optimizer = self._build_optimizer()

    def _build_generator(self):
        """
        建構生成器線路 (Generator Circuit)
        使用 EfficientSU2 增加電路深度 (Depth) 以捕捉複雜分佈
        """
        circuit = EfficientSU2(
            self.num_qubits, 
            su2_gates=['ry', 'rz'], 
            entanglement='full', 
            reps=self.reps
        )
        return circuit

    def _build_optimizer(self):
        """
        建構 SPSA 最佳化函數
        適合 NISQ 含噪環境的無梯度優化方法
        """
        # perturbation 與 learning_rate 可依實驗調整
        return SPSA(maxiter=500, learning_rate=0.01, perturbation=0.05)

    def get_info(self):
        return {
            "num_qubits": self.num_qubits,
            "num_parameters": self.generator_circuit.num_parameters,
            "optimizer": "SPSA",
            "circuit_type": "EfficientSU2 (Full Entanglement)"
        }

if __name__ == "__main__":
    qgan = QuantumDataGenerator()
    print("QGAN Generator Config:", qgan.get_info())
    print("Circuit Topology:")
    # qgan.generator_circuit.draw('mpl') # 需要在 Notebook 中執行才能看到圖