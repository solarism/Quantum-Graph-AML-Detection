import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    計算反洗錢偵測的關鍵指標 [Source 112]
    """
    y_pred_bin = (y_pred > threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    
    # 計算誤報率 (False Positive Rate) [Source 117]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {"F1": f1, "AUC": auc, "FPR": fpr}

def plot_tsne(features, labels, save_path="results/figures/tsne_plot.png"):
    """
    繪製 t-SNE 降維圖，視覺化量子特徵的分離度 [Source 127]
    """
    print("Generating t-SNE plot...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    # 繪製合法交易 (Label 0)
    plt.scatter(embeddings_2d[labels==0, 0], embeddings_2d[labels==0, 1], 
                c='blue', label='Licit', alpha=0.5, s=10)
    # 繪製非法洗錢交易 (Label 1)
    plt.scatter(embeddings_2d[labels==1, 0], embeddings_2d[labels==1, 1], 
                c='red', label='Illicit', alpha=0.6, s=10)
    
    plt.title("t-SNE Visualization of Transaction Embeddings")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")