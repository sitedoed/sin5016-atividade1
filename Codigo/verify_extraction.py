import os
import numpy as np
import json
from collections import Counter

def verify_extraction(features_dir):
    """Verifica os dados extraidos"""
    
    print("="*60)
    print("VERIFICACAO DOS DADOS EXTRAIDOS")
    print("="*60)
    
    # Verificar arquivos
    required_files = [
        ('hog/features.npy', 'Caracteristicas HOG'),
        ('lbp/features.npy', 'Caracteristicas LBP'),
        ('combined/features.npy', 'Caracteristicas combinadas'),
        ('metadata/labels.npy', 'Labels'),
        ('metadata/original_labels.npy', 'Labels originais'),
        ('metadata/label_mapping.json', 'Mapeamento de labels')
    ]
    
    all_exist = True
    for file, description in required_files:
        path = os.path.join(features_dir, file)
        if os.path.exists(path):
            print(f"OK  {description}: {os.path.basename(path)}")
        else:
            print(f"ERRO {description}: FALTANDO")
            all_exist = False
    
    if not all_exist:
        print("\nALGUNS ARQUIVOS ESTAO FALTANDO!")
        return
    
    # Carregar dados
    print("\nCarregando dados...")
    
    hog_features = np.load(os.path.join(features_dir, 'hog', 'features.npy'))
    lbp_features = np.load(os.path.join(features_dir, 'lbp', 'features.npy'))
    labels = np.load(os.path.join(features_dir, 'metadata', 'labels.npy'))
    original_labels = np.load(os.path.join(features_dir, 'metadata', 'original_labels.npy'))
    
    # Carregar mapeamento
    with open(os.path.join(features_dir, 'metadata', 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    
    # Estatisticas basicas
    print(f"\nEstatisticas:")
    print(f"   Total de amostras: {len(labels)}")
    print(f"   Classes unicas: {len(np.unique(labels))}")
    print(f"   Dimensoes HOG: {hog_features.shape}")
    print(f"   Dimensoes LBP: {lbp_features.shape}")
    
    # Distribuicao de classes
    label_counts = Counter(labels)
    print(f"\nDistribuicao de classes:")
    print(f"   Media de amostras por classe: {np.mean(list(label_counts.values())):.2f}")
    print(f"   Desvio padrao: {np.std(list(label_counts.values())):.2f}")
    print(f"   Minimo: {min(label_counts.values())}")
    print(f"   Maximo: {max(label_counts.values())}")
    
    # Verificar valores
    print(f"\nVerificacao de valores:")
    print(f"   HOG - NaN: {np.isnan(hog_features).sum()}, Inf: {np.isinf(hog_features).sum()}")
    print(f"   LBP - NaN: {np.isnan(lbp_features).sum()}, Inf: {np.isinf(lbp_features).sum()}")
    
    # Verificar normalizacao
    print(f"\nVerificacao de normalizacao:")
    print(f"   HOG - Media: {hog_features.mean():.6f}, Desvio: {hog_features.std():.6f}")
    print(f"   LBP - Media: {lbp_features.mean():.6f}, Desvio: {lbp_features.std():.6f}")
    
    # Exemplos
    print(f"\nExemplos de mapeamento de labels:")
    unique_numeric_labels = np.unique(labels)[:5]
    for num_label in unique_numeric_labels:
        idx = np.where(labels == num_label)[0][0]
        original = original_labels[idx]
        print(f"   Label numerico {num_label} -> Original: {original}")
    
    print(f"\nVERIFICACAO CONCLUIDA COM SUCESSO!")

if __name__ == "__main__":
    features_dir = "../data/features"
    verify_extraction(features_dir)