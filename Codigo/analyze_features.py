import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

def analyze_features(features_dir):
    """
    Realiza análise exploratória das características extraídas
    """
    # Carregar dados
    hog_features = np.load(os.path.join(features_dir, 'hog', 'features.npy'))
    lbp_features = np.load(os.path.join(features_dir, 'lbp', 'features.npy'))
    labels = np.load(os.path.join(features_dir, 'metadata', 'labels.npy'))
    
    print("="*60)
    print("ANÁLISE EXPLORATÓRIA DAS CARACTERÍSTICAS")
    print("="*60)
    
    # Estatísticas básicas
    print(f"\n📊 Estatísticas Básicas:")
    print(f"   Total de amostras: {len(labels)}")
    print(f"   Classes únicas: {len(np.unique(labels))}")
    print(f"   Dimensões HOG: {hog_features.shape}")
    print(f"   Dimensões LBP: {lbp_features.shape}")
    
    # Distribuição de classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Plotar distribuição de classes (top 20)
    plt.figure(figsize=(12, 6))
    
    # Top 20 classes com mais amostras
    sorted_indices = np.argsort(counts)[-20:]
    top_labels = unique_labels[sorted_indices]
    top_counts = counts[sorted_indices]
    
    plt.subplot(1, 2, 1)
    plt.barh(range(len(top_labels)), top_counts)
    plt.yticks(range(len(top_labels)), top_labels)
    plt.xlabel('Número de Imagens')
    plt.title('Top 20 Pessoas com Mais Imagens')
    plt.grid(axis='x', alpha=0.3)
    
    # Histograma de distribuição
    plt.subplot(1, 2, 2)
    plt.hist(counts, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Número de Imagens por Pessoa')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Imagens por Pessoa')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(features_dir, '..', '..', 'results', 'plots', 'class_distribution.png'), dpi=150)
    plt.show()
    
    # Visualização t-SNE para HOG features
    print("\n🔍 Redução de dimensionalidade (t-SNE)...")
    
    # Amostrar dados para t-SNE (muito intensivo computacionalmente)
    n_samples = min(1000, len(labels))
    indices = np.random.choice(len(labels), n_samples, replace=False)
    
    # Aplicar PCA primeiro para reduzir dimensionalidade
    pca = PCA(n_components=50)
    hog_pca = pca.fit_transform(hog_features[indices])
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    hog_tsne = tsne.fit_transform(hog_pca)
    
    # Plotar t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(hog_tsne[:, 0], hog_tsne[:, 1], 
                         c=pd.factorize(labels[indices])[0], 
                         cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title('Visualização t-SNE das Características HOG')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(os.path.join(features_dir, '..', '..', 'results', 'plots', 'tsne_hog.png'), dpi=150)
    plt.show()
    
    # Matriz de correlação das características HOG (primeiras 20)
    print("\n📈 Matriz de correlação das características HOG...")
    
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(hog_features[:100, :20].T)  # Primeiras 20 features de 100 amostras
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlação (Primeiras 20 Features HOG)')
    plt.tight_layout()
    plt.savefig(os.path.join(features_dir, '..', '..', 'results', 'plots', 'correlation_matrix.png'), dpi=150)
    plt.show()
    
    # Histogramas de valores de características
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograma HOG
    axes[0, 0].hist(hog_features.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribuição dos Valores HOG')
    axes[0, 0].set_xlabel('Valor')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].grid(alpha=0.3)
    
    # Histograma LBP
    axes[0, 1].hist(lbp_features.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Distribuição dos Valores LBP')
    axes[0, 1].set_xlabel('Valor')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].grid(alpha=0.3)
    
    # Boxplot HOG por classe (primeiras 5 classes)
    axes[1, 0].boxplot([hog_features[labels == label][:100, 0] for label in unique_labels[:5]])
    axes[1, 0].set_xticklabels(unique_labels[:5], rotation=45)
    axes[1, 0].set_title('Boxplot da Primeira Feature HOG por Classe')
    axes[1, 0].set_ylabel('Valor da Feature')
    axes[1, 0].grid(alpha=0.3)
    
    # Variância explicada por PCA
    pca_full = PCA().fit(hog_features)
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    axes[1, 1].plot(explained_variance, marker='o')
    axes[1, 1].set_xlabel('Número de Componentes PCA')
    axes[1, 1].set_ylabel('Variância Acumulada Explicada')
    axes[1, 1].set_title('Variância Explicada por Componentes PCA')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(features_dir, '..', '..', 'results', 'plots', 'feature_analysis.png'), dpi=150)
    plt.show()
    
    # Relatório final
    print("\n" + "="*60)
    print("RELATÓRIO DA ANÁLISE")
    print("="*60)
    
    # Calcular informações importantes
    n_components_95 = np.argmax(explained_variance >= 0.95) + 1
    print(f"\n📋 Informações para modelagem:")
    print(f"   Componentes PCA para 95% variância: {n_components_95}")
    print(f"   Dimensão original HOG: {hog_features.shape[1]}")
    print(f"   Redução possível: {hog_features.shape[1] / n_components_95:.1f}x")
    
    # Verificar balanceamento
    imbalance_ratio = counts.max() / counts.min()
    print(f"\n⚖️ Balanceamento das classes:")
    print(f"   Razão de desbalanceamento: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print("   ⚠️  Atenção: Classes muito desbalanceadas!")
    elif imbalance_ratio > 5:
        print("   ⚠️  Classes moderadamente desbalanceadas")
    else:
        print("   ✓ Classes razoavelmente balanceadas")
    
    # Salvar relatório
    report_path = os.path.join(features_dir, '..', '..', 'results', 'logs', 'feature_analysis_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE ANÁLISE DE CARACTERÍSTICAS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total de amostras: {len(labels)}\n")
        f.write(f"Classes únicas: {len(unique_labels)}\n")
        f.write(f"Dimensões HOG: {hog_features.shape}\n")
        f.write(f"Dimensões LBP: {lbp_features.shape}\n")
        f.write(f"\nDistribuição de classes:\n")
        for label, count in zip(unique_labels[:10], counts[:10]):
            f.write(f"  {label}: {count} imagens\n")
        f.write(f"\nComponentes PCA para 95% variância: {n_components_95}\n")
        f.write(f"Razão de desbalanceamento: {imbalance_ratio:.2f}\n")

if __name__ == "__main__":
    # Definir caminho
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(base_dir, "data", "features")
    
    if not os.path.exists(features_dir):
        print(f"Erro: Diretório de características não encontrado: {features_dir}")
        print("Execute primeiro extract_features.py")
    else:
        analyze_features(features_dir)