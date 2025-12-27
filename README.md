# Quantum-Graph-AML-Detection 威震＠NTUB

![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Qiskit-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Project Abstract (研究摘要)

This project implements a **Hybrid Quantum-Classical Graph Neural Network (Hybrid QGNN)** designed to detect illicit cryptocurrency transactions (Anti-Money Laundering, AML) on blockchain networks. 

Targeting the **NISQ (Noisy Intermediate-Scale Quantum)** era constraints, we propose a dual-stage architecture:
1.  **Classical Encoder**: Uses **GraphSAGE** to handle dynamic graph structures and solve the "over-smoothing" problem in deep GNNs.
2.  **Quantum Kernel**: Utilizes **ZZ-Feature Maps** and Variational Quantum Circuits (VQC) to map low-dimensional features into high-dimensional Hilbert Space, enhancing the separability of illicit transactions (typically <2% of data).

This repository serves as the Proof-of-Concept (PoC) implementation for the research proposal.

---

## 🏗️ System Architecture (系統架構)

The architecture is designed to minimize quantum noise impact while maximizing feature extraction capabilities:

```mermaid
graph LR
    A[Raw Blockchain Data] --> B[Data Preprocessing];
    B --> C[Classical GNN Layer];
    C --> D[Dimension Reduction];
    D --> E[Quantum Feature Map];
    E --> F[Variational Circuit];
    F --> G[Risk Prediction];
    
    subgraph Classical [Module 1 & 2: Classical Optimization]
    B(Temporal Split & SMOTE)
    C(GraphSAGE Encoder)
    D(Linear Compression)
    end
    
    subgraph Quantum [Module 3: Quantum Kernel]
    E(ZZ-Feature Map)
    F(Ansatz / PQC)
    end