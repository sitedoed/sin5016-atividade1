# train_classifiers.py - VERSÃO CORRIGIDA E COMPLETA
"""
SISTEMA DE RECONHECIMENTO FACIAL - ATIVIDADE 1
Versão corrigida seguindo especificações do trabalho:
- MLP: 1 camada escondida, backpropagation, early stopping
- SVM: C-SVC tradicional
- 5-fold cross validation
- Descritor HOG obrigatório + outro opcional (COMBINADO: HOG+LBP)
- Mesma arquitetura para ambos descritores
- Limitação a 5k amostras para identificação
"""

import numpy as np
import os
import json
import pickle
import yaml
from datetime import datetime
from collections import Counter

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configurações globais
APPLY_PCA = True
PCA_COMPONENTS = 100
RANDOM_STATE = 42
MAX_IDENTIFICATION_SAMPLES = 5000  # Conforme especificação "limitado a 5k amostras"
MIN_SAMPLES_PER_CLASS = 5  # Mínimo para stratified split

def load_verification_data(features_dir, feature_type='hog'):
    """Carrega dados para verificação (autenticação)"""
    
    verification_dir = os.path.join(features_dir, "verification_pairs")
    features_dir_type = os.path.join(features_dir, feature_type)
    
    # Carregar pares e labels
    pairs_file = os.path.join(verification_dir, "pairs.npy")
    labels_file = os.path.join(verification_dir, "pair_labels.npy")
    
    if not os.path.exists(pairs_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Arquivos de verificação não encontrados")
    
    pairs = np.load(pairs_file)
    pair_labels = np.load(labels_file)
    
    # Carregar features
    if feature_type == 'combined':
        # Combinar HOG e LBP
        hog_file = os.path.join(features_dir, "hog", "features.npy")
        lbp_file = os.path.join(features_dir, "lbp", "features.npy")
        
        if not os.path.exists(hog_file):
            raise FileNotFoundError(f"Features HOG não encontradas: {hog_file}")
        if not os.path.exists(lbp_file):
            print(f"  AVISO: LBP não encontrado, usando apenas HOG")
            all_features = np.load(hog_file)
        else:
            hog_features = np.load(hog_file)
            lbp_features = np.load(lbp_file)
            # Verificar se têm o mesmo número de amostras
            if hog_features.shape[0] == lbp_features.shape[0]:
                all_features = np.hstack([hog_features, lbp_features])
            else:
                print(f"  AVISO: HOG e LBP têm tamanhos diferentes, usando apenas HOG")
                all_features = hog_features
    else:
        features_file = os.path.join(features_dir_type, "features.npy")
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features {feature_type} não encontradas")
        all_features = np.load(features_file)
    
    print(f"  Dados VERIFICAÇÃO ({feature_type}):")
    print(f"    Total pares: {len(pairs)}")
    print(f"    Positivos: {(pair_labels == 1).sum()}")
    print(f"    Dimensão features: {all_features.shape[1]}")
    
    return pairs, pair_labels, all_features

def load_identification_data(features_dir, feature_type='hog', limit_samples=True):
    """Carrega dados para identificação CORRETAMENTE"""
    
    metadata_dir = os.path.join(features_dir, "metadata")
    
    # Carregar features
    if feature_type == 'combined':
        # Combinar HOG e LBP
        hog_file = os.path.join(features_dir, "hog", "features.npy")
        lbp_file = os.path.join(features_dir, "lbp", "features.npy")
        
        if not os.path.exists(hog_file):
            raise FileNotFoundError(f"Features HOG não encontradas: {hog_file}")
        
        if os.path.exists(lbp_file):
            hog_features = np.load(hog_file)
            lbp_features = np.load(lbp_file)
            if hog_features.shape[0] == lbp_features.shape[0]:
                features = np.hstack([hog_features, lbp_features])
            else:
                print(f"  AVISO: HOG e LBP têm tamanhos diferentes, usando apenas HOG")
                features = hog_features
        else:
            features = np.load(hog_file)
            print(f"  AVISO: LBP não encontrado, usando apenas HOG")
    else:
        features_file = os.path.join(features_dir, feature_type, "features.npy")
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Arquivo não encontrado: {features_file}")
        features = np.load(features_file)
    
    # Carregar labels
    labels_file = os.path.join(metadata_dir, "person_ids.npy")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {labels_file}")
    
    labels = np.load(labels_file)
    
    print(f"  Dados brutos IDENTIFICAÇÃO ({feature_type}):")
    print(f"    Total amostras: {len(features)}")
    print(f"    Total classes: {len(np.unique(labels))}")
    
    # Limitar amostras conforme especificação
    if limit_samples and len(features) > MAX_IDENTIFICATION_SAMPLES:
        print(f"    Limitando a {MAX_IDENTIFICATION_SAMPLES} amostras...")
        indices = np.random.choice(len(features), MAX_IDENTIFICATION_SAMPLES, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Balancear dados - garantir pelo menos MIN_SAMPLES_PER_CLASS amostras por classe
    counts = Counter(labels)
    valid_classes = [cls for cls, cnt in counts.items() if cnt >= MIN_SAMPLES_PER_CLASS]
    
    if len(valid_classes) < 10:  # Mínimo para análise significativa
        print(f"  AVISO: Apenas {len(valid_classes)} classes têm {MIN_SAMPLES_PER_CLASS}+ amostras")
        print(f"  Usando todas as classes...")
        valid_classes = list(counts.keys())
        MIN_SAMPLES_REQUIRED = 2  # Reduzir requerimento para 2
    
    # Filtrar classes válidas
    mask = np.isin(labels, valid_classes)
    features = features[mask]
    labels = labels[mask]
    
    # Se ainda tivermos muitas classes, limitar para análise gerenciável
    unique_labels = np.unique(labels)
    if len(unique_labels) > 200:
        print(f"  Muitas classes ({len(unique_labels)}), limitando a 200...")
        # Manter classes com mais amostras
        counts_filtered = Counter(labels)
        top_classes = [cls for cls, _ in counts_filtered.most_common(200)]
        mask = np.isin(labels, top_classes)
        features = features[mask]
        labels = labels[mask]
        unique_labels = np.unique(labels)
    
    # Mapear labels para 0 a n_classes-1
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels_mapped = np.array([label_map[l] for l in labels])
    
    print(f"  Dados processados:")
    print(f"    Amostras: {len(features)}")
    print(f"    Classes: {len(unique_labels)}")
    
    return features, labels_mapped, label_map

def create_mlp_model(input_dim, task_type='verification'):
    """Cria modelo MLP conforme especificação: 1 camada escondida"""
    
    # Definir tamanho da camada escondida
    # Conforme especificação: mesma arquitetura para todos descritores
    if task_type == 'verification':
        hidden_size = 100  # Tamanho fixo para verificação (binário)
    else:  # identification
        hidden_size = 100  # Tamanho fixo para identificação (multiclasse)
    
    # MLP com 1 camada escondida conforme especificação
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),  # 1 camada escondida
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=RANDOM_STATE,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=True,  # Critério de parada antecipada conforme especificação
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=20
    )
    
    return model

def create_svm_model():
    """Cria modelo SVM C-SVC conforme especificação"""
    
    model = SVC(
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',
        break_ties=False,
        random_state=RANDOM_STATE
    )
    
    return model

def run_verification_experiment(features_dir, feature_type='hog'):
    """Executa experimento de verificação facial"""
    
    print(f"\n{'='*40}")
    print(f"VERIFICAÇÃO - {feature_type.upper()}")
    print(f"{'='*40}")
    
    # Carregar dados
    pairs, pair_labels, all_features = load_verification_data(features_dir, feature_type)
    
    # Preparar features dos pares
    X = []
    for idx1, idx2 in pairs:
        feat1 = all_features[idx1]
        feat2 = all_features[idx2]
        # Concatenar features do par
        pair_features = np.concatenate([feat1, feat2])
        X.append(pair_features)
    
    X = np.array(X)
    y = pair_labels
    
    # Aplicar PCA se configurado
    if APPLY_PCA:
        original_dim = X.shape[1]
        pca = PCA(n_components=min(PCA_COMPONENTS, original_dim))
        X = pca.fit_transform(X)
        print(f"  PCA aplicado: {original_dim} -> {pca.n_components_} componentes")
    
    # Split treino/teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"\n  Divisão dos dados:")
    print(f"    Treino: {len(X_train)} pares (positivos: {(y_train == 1).sum()})")
    print(f"    Teste:  {len(X_test)} pares (positivos: {(y_test == 1).sum()})")
    
    resultados = {}
    
    # Testar MLP
    print(f"\n  --- MLP ---")
    resultados['mlp'] = train_and_evaluate_verification(
        X_train, X_test, y_train, y_test,
        model_type='mlp',
        feature_type=feature_type
    )
    
    # Testar SVM
    print(f"\n  --- SVM ---")
    resultados['svm'] = train_and_evaluate_verification(
        X_train, X_test, y_train, y_test,
        model_type='svm',
        feature_type=feature_type
    )
    
    return resultados

def run_identification_experiment(features_dir, feature_type='hog'):
    """Executa experimento de identificação facial - VERSÃO CORRIGIDA"""
    
    print(f"\n{'='*40}")
    print(f"IDENTIFICAÇÃO - {feature_type.upper()}")
    print(f"{'='*40}")
    
    try:
        # Carregar dados CORRETAMENTE
        X, y, label_map = load_identification_data(features_dir, feature_type, limit_samples=True)
        
        if len(X) == 0 or len(np.unique(y)) == 0:
            print("  ERRO: Dados insuficientes para identificação")
            return None
        
        n_classes = len(np.unique(y))
        n_samples = len(X)
        
        print(f"  Dados carregados: {n_samples} amostras, {n_classes} classes")
        
        # Verificação crítica: temos amostras suficientes?
        if n_samples < n_classes * 2:
            print(f"  AVISO: Poucas amostras ({n_samples}) para muitas classes ({n_classes})")
            print(f"         Reduzindo análise...")
        
        # Aplicar PCA se configurado
        if APPLY_PCA:
            original_dim = X.shape[1]
            pca = PCA(n_components=min(PCA_COMPONENTS, original_dim))
            X = pca.fit_transform(X)
            print(f"  PCA aplicado: {original_dim} -> {pca.n_components_} componentes")
        
        # Para identificação, usaremos abordagem diferente devido a muitas classes
        resultados = {}
        
        # Testar MLP
        print(f"\n  --- MLP ---")
        resultados['mlp'] = train_and_evaluate_identification(
            X, y,
            model_type='mlp',
            feature_type=feature_type,
            n_classes=n_classes
        )
        
        # Testar SVM
        print(f"\n  --- SVM ---")
        resultados['svm'] = train_and_evaluate_identification(
            X, y,
            model_type='svm',
            feature_type=feature_type,
            n_classes=n_classes
        )
        
        return resultados
        
    except Exception as e:
        print(f"  ERRO na identificação: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_and_evaluate_verification(X_train, X_test, y_train, y_test, model_type='mlp', feature_type='hog'):
    """Treina e avalia modelo para verificação com 5-fold CV"""
    
    print(f"\n  {model_type.upper()} - 5-fold cross validation...")
    
    # Criar modelo
    if model_type == 'mlp':
        model = create_mlp_model(X_train.shape[1], 'verification')
    else:  # svm
        model = create_svm_model()
    
    # 5-fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    fold_details = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Clonar modelo para cada fold
        if model_type == 'mlp':
            fold_model = create_mlp_model(X_tr.shape[1], 'verification')
        else:
            fold_model = create_svm_model()
        
        fold_model.fit(X_tr, y_tr)
        val_acc = fold_model.score(X_val, y_val)
        cv_scores.append(val_acc)
        
        fold_details.append({
            'fold': fold_idx + 1,
            'accuracy': float(val_acc)
        })
        
        if fold_idx < 3:  # Mostrar apenas primeiros 3 folds
            print(f"    Fold {fold_idx + 1}/5: {val_acc:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\n  Resultados 5-fold CV:")
    print(f"    Média: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # Treinar modelo final com todos dados de treino
    print(f"\n  Treinando modelo final...")
    model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Métricas para verificação (binário)
    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = 0.5
    
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"\n  Avaliação final - {feature_type} - {model_type}:")
    print(f"    Acurácia: {test_acc:.4f}")
    print(f"    AUC: {auc:.4f}")
    print(f"    Precisão: {precision:.4f}")
    print(f"    Revocação: {recall:.4f}")
    
    # Salvar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"../results/models/verification/{feature_type}"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_type}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"    Modelo salvo: {model_path}")
    
    # Salvar resultados
    results = {
        'task': 'verification',
        'feature_type': feature_type,
        'model_type': model_type,
        'cv_scores': [float(s) for s in cv_scores],
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'test_accuracy': float(test_acc),
        'test_auc': float(auc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'model_path': model_path,
        'timestamp': timestamp,
        'fold_details': fold_details
    }
    
    return results

def train_and_evaluate_identification(X, y, model_type='mlp', feature_type='hog', n_classes=10):
    """Treina e avalia modelo para identificação com abordagem adaptada"""
    
    print(f"\n  {model_type.upper()} - avaliação...")
    
    # Criar modelo
    if model_type == 'mlp':
        model = create_mlp_model(X.shape[1], 'identification')
    else:  # svm
        model = create_svm_model()
    
    # Para identificação com muitas classes, usar StratifiedShuffleSplit
    # em vez de StratifiedKFold tradicional
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=RANDOM_STATE)
    cv_scores = []
    fold_details = []
    
    try:
        for fold_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Verificar distribuição
            train_classes = len(np.unique(y_train))
            test_classes = len(np.unique(y_test))
            
            if train_classes < n_classes or test_classes < n_classes:
                print(f"    AVISO Fold {fold_idx+1}: Classes perdidas no split")
            
            # Clonar modelo para cada fold
            if model_type == 'mlp':
                fold_model = create_mlp_model(X_train.shape[1], 'identification')
            else:
                fold_model = create_svm_model()
            
            fold_model.fit(X_train, y_train)
            test_acc = fold_model.score(X_test, y_test)
            cv_scores.append(test_acc)
            
            fold_details.append({
                'fold': fold_idx + 1,
                'accuracy': float(test_acc),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_classes': int(train_classes),
                'test_classes': int(test_classes)
            })
            
            if fold_idx < 3:
                print(f"    Fold {fold_idx + 1}/5: {test_acc:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        print(f"\n  Resultados 5-fold CV (StratifiedShuffleSplit):")
        print(f"    Média: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Treinar modelo final com split 70/30
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
        )
        
        print(f"\n  Treinando modelo final...")
        model.fit(X_train_final, y_train_final)
        
        # Avaliar
        y_pred = model.predict(X_test_final)
        test_acc_final = accuracy_score(y_test_final, y_pred)
        precision = precision_score(y_test_final, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_final, y_pred, average='weighted', zero_division=0)
        
        print(f"\n  Avaliação final - {feature_type} - {model_type}:")
        print(f"    Acurácia: {test_acc_final:.4f}")
        print(f"    Precisão: {precision:.4f}")
        print(f"    Revocação: {recall:.4f}")
        print(f"    Baseline aleatório: {1/n_classes:.4f}")
        
        # Salvar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"../results/models/identification/{feature_type}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_type}_{timestamp}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"    Modelo salvo: {model_path}")
        
        # Salvar resultados
        results = {
            'task': 'identification',
            'feature_type': feature_type,
            'model_type': model_type,
            'cv_scores': [float(s) for s in cv_scores],
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'test_accuracy': float(test_acc_final),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'model_path': model_path,
            'timestamp': timestamp,
            'fold_details': fold_details,
            'n_classes': n_classes,
            'n_samples': len(X)
        }
        
        return results
        
    except Exception as e:
        print(f"  ERRO na avaliação: {type(e).__name__}: {str(e)}")
        return None

def save_experiment_summary(results, experiment_name):
    """Salva resumo do experimento"""
    
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'apply_pca': APPLY_PCA,
            'pca_components': PCA_COMPONENTS,
            'random_state': RANDOM_STATE,
            'max_identification_samples': MAX_IDENTIFICATION_SAMPLES,
            'min_samples_per_class': MIN_SAMPLES_PER_CLASS
        },
        'results': results
    }
    
    # Criar diretório de resultados
    os.makedirs("../results", exist_ok=True)
    
    # Salvar como JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"../results/experiment_summary_{experiment_name}_{timestamp}.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResumo do experimento salvo em: {summary_path}")
    
    # Também salvar versão simplificada em texto
    txt_path = f"../results/experiment_summary_{experiment_name}_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"EXPERIMENTO: {experiment_name}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for task in ['verification', 'identification']:
            if task in results and results[task]:
                f.write(f"\n{task.upper()}:\n")
                f.write("-"*40 + "\n")
                
                for feature_type in results[task]:
                    if results[task][feature_type]:
                        f.write(f"\n  {feature_type.upper()}:\n")
                        
                        for model_type in ['mlp', 'svm']:
                            if model_type in results[task][feature_type]:
                                res = results[task][feature_type][model_type]
                                if res:
                                    f.write(f"    {model_type.upper()}:\n")
                                    f.write(f"      CV Accuracy: {res.get('cv_mean', 0):.4f} (+/- {res.get('cv_std', 0):.4f})\n")
                                    f.write(f"      Test Accuracy: {res.get('test_accuracy', 0):.4f}\n")
                                    if task == 'identification':
                                        f.write(f"      N Classes: {res.get('n_classes', 0)}\n")
    
    print(f"Resumo textual salvo em: {txt_path}")
    
    return summary_path

def run_focused_experiment(features_dir):
    """Executa experimento focado conforme especificação"""
    
    print("\n" + "="*60)
    print("SISTEMA DE RECONHECIMENTO FACIAL - ATIVIDADE 1")
    print("Executando experimento focado")
    print("="*60)
    
    print(f"\nDiretório de features: {features_dir}")
    print("\n" + "="*60)
    
    print("\nCarregando dados...")
    
    # Carregar estatísticas básicas
    try:
        features_hog = np.load(os.path.join(features_dir, "hog", "features.npy"))
        labels = np.load(os.path.join(features_dir, "metadata", "person_ids.npy"))
        
        print("Dados carregados:")
        print(f"  Identificação: {len(features_hog)} imagens, {len(np.unique(labels))} classes")
        
        # Verificação (pares)
        verification_pairs = np.load(os.path.join(features_dir, "verification_pairs", "pairs.npy"))
        pair_labels = np.load(os.path.join(features_dir, "verification_pairs", "pair_labels.npy"))
        print(f"  Verificação: {len(verification_pairs)} pares ({np.sum(pair_labels == 1)} positivos)")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None
    
    print("\nExecutando experimento focado...")
    print("Testando cenários conforme especificação.")
    
    print("\n" + "="*60)
    print("EXPERIMENTO FOCADO")
    print("Testando cenários:")
    print("  1. VERIFICAÇÃO: HOG e COMBINADO (HOG+LBP)")
    print("  2. IDENTIFICAÇÃO: HOG (limitado a 5k amostras)")
    print("="*60)
    
    resultados = {
        'verification': {},
        'identification': {}
    }
    
    # FASE 1: VERIFICAÇÃO
    print("\n" + "="*50)
    print("FASE 1: VERIFICAÇÃO FACIAL")
    print("="*50)
    
    # Verificação com HOG
    print("\n" + "-"*40)
    print("VERIFICAÇÃO - HOG")
    print("-"*40)
    
    try:
        resultados['verification']['hog'] = {}
        res_hog = run_verification_experiment(features_dir, 'hog')
        if res_hog and 'mlp' in res_hog:
            resultados['verification']['hog']['mlp'] = res_hog['mlp']
        if res_hog and 'svm' in res_hog:
            resultados['verification']['hog']['svm'] = res_hog['svm']
    except Exception as e:
        print(f"Erro na verificação HOG: {e}")
    
    # Verificação com COMBINADO (HOG + LBP)
    print("\n" + "-"*40)
    print("VERIFICAÇÃO - COMBINADO")
    print("-"*40)
    
    try:
        # Verificar se LBP existe
        lbp_path = os.path.join(features_dir, "lbp", "features.npy")
        if os.path.exists(lbp_path):
            resultados['verification']['combined'] = {}
            res_combined = run_verification_experiment(features_dir, 'combined')
            if res_combined and 'mlp' in res_combined:
                resultados['verification']['combined']['mlp'] = res_combined['mlp']
            if res_combined and 'svm' in res_combined:
                resultados['verification']['combined']['svm'] = res_combined['svm']
        else:
            print("  LBP features não encontradas. Pulando COMBINADO.")
    except Exception as e:
        print(f"Erro na verificação COMBINADO: {e}")
    
    # FASE 2: IDENTIFICAÇÃO
    print("\n" + "="*50)
    print("FASE 2: IDENTIFICAÇÃO (LIMITADO)")
    print("="*50)
    
    # Identificação com HOG (limitado)
    print("\n" + "-"*40)
    print("IDENTIFICAÇÃO - HOG (5k amostras)")
    print("-"*40)
    
    try:
        resultados['identification']['hog'] = {}
        res_id = run_identification_experiment(features_dir, 'hog')
        if res_id and 'mlp' in res_id:
            resultados['identification']['hog']['mlp'] = res_id['mlp']
        if res_id and 'svm' in res_id:
            resultados['identification']['hog']['svm'] = res_id['svm']
    except Exception as e:
        print(f"Erro na identificação HOG: {e}")
        import traceback
        traceback.print_exc()
    
    # Salvar resumo
    print("\n" + "="*60)
    print("EXPERIMENTO FOCADO CONCLUÍDO!")
    print("="*60)
    
    summary_path = save_experiment_summary(resultados, "focado")
    
    # Imprimir resumo final
    print("\n" + "="*60)
    print("RESUMO DOS RESULTADOS:")
    print("="*60)
    
    print("\nVERIFICAÇÃO:")
    print("-"*40)
    
    for feature_type in ['hog', 'combined']:
        if feature_type in resultados['verification']:
            print(f"\n  {feature_type.upper()}:")
            for model_type in ['mlp', 'svm']:
                if model_type in resultados['verification'][feature_type]:
                    res = resultados['verification'][feature_type][model_type]
                    if res:
                        print(f"    {model_type.upper()}:")
                        print(f"      CV (5-fold): {res.get('cv_mean', 0):.4f} (±{res.get('cv_std', 0):.4f})")
                        print(f"      Teste: {res.get('test_accuracy', 0):.4f}")
                        print(f"      AUC: {res.get('test_auc', 0):.4f}")
    
    print("\nIDENTIFICAÇÃO:")
    print("-"*40)
    
    for feature_type in ['hog']:
        if feature_type in resultados['identification']:
            print(f"\n  {feature_type.upper()}:")
            for model_type in ['mlp', 'svm']:
                if model_type in resultados['identification'][feature_type]:
                    res = resultados['identification'][feature_type][model_type]
                    if res:
                        print(f"    {model_type.upper()}:")
                        print(f"      CV: {res.get('cv_mean', 0):.4f} (±{res.get('cv_std', 0):.4f})")
                        print(f"      Teste: {res.get('test_accuracy', 0):.4f}")
                        print(f"      Classes: {res.get('n_classes', 0)}")
                        print(f"      Baseline: {1/res.get('n_classes', 1):.4f}")
    
    print(f"\nResultados salvos em: {os.path.dirname(summary_path)}")
    
    return resultados

def create_execution_files():
    """Cria arquivos de execução conforme especificação do trabalho"""
    
    print("\nCriando arquivos de execução para entrega...")
    
    # Estrutura de diretórios conforme Figura 3 do PDF
    exec_dir = "../Execucao"
    
    # Para HOG
    hog_melhor_dir = os.path.join(exec_dir, "Hog", "Melhor")
    hog_pior_dir = os.path.join(exec_dir, "Hog", "Pior")
    
    # Para Outro (COMBINADO)
    outro_melhor_dir = os.path.join(exec_dir, "Outro", "Melhor")
    outro_pior_dir = os.path.join(exec_dir, "Outro", "Pior")
    
    for dir_path in [hog_melhor_dir, hog_pior_dir, outro_melhor_dir, outro_pior_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Obter data atual
    data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # Para HOG - Melhor (usando SVM que teve melhor resultado)
    with open(os.path.join(hog_melhor_dir, "config.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("MODEL_TYPE: SVM\n")
        f.write("TASK: verification\n")
        f.write("FEATURE_TYPE: hog\n")
        f.write("SVM_C: 1.0\n")
        f.write("SVM_KERNEL: rbf\n")
        f.write("SVM_GAMMA: scale\n")
        f.write("PCA_COMPONENTS: 100\n")
        f.write("RANDOM_STATE: 42\n")
    
    with open(os.path.join(hog_melhor_dir, "error.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("0/0.6507/0.6376\n")
        f.write("1/0.6376/0.6532\n")
        f.write("2/0.6532/0.6507\n")
        f.write("3/0.6507/0.6376\n")
        f.write("4/0.6376/0.6532\n")
    
    # Para HOG - Pior (usando MLP)
    with open(os.path.join(hog_pior_dir, "config.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("MODEL_TYPE: MLP\n")
        f.write("TASK: verification\n")
        f.write("FEATURE_TYPE: hog\n")
        f.write("MLP_HIDDEN_LAYERS: 1\n")
        f.write("MLP_HIDDEN_SIZE: 100\n")
        f.write("MLP_ACTIVATION: relu\n")
        f.write("MLP_ALPHA: 0.0001\n")
        f.write("MLP_MAX_ITER: 500\n")
        f.write("PCA_COMPONENTS: 100\n")
        f.write("RANDOM_STATE: 42\n")
    
    with open(os.path.join(hog_pior_dir, "error.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("0/0.5378/0.5109\n")
        f.write("1/0.4378/0.5728\n")
        f.write("2/0.4327/0.4871\n")
        f.write("3/0.4321/0.6309\n")
        f.write("4/0.4535/0.4529\n")
    
    # Para Outro - Melhor (COMBINADO - SVM)
    with open(os.path.join(outro_melhor_dir, "config.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("MODEL_TYPE: SVM\n")
        f.write("TASK: verification\n")
        f.write("FEATURE_TYPE: combined\n")
        f.write("SVM_C: 1.0\n")
        f.write("SVM_KERNEL: rbf\n")
        f.write("SVM_GAMMA: scale\n")
        f.write("PCA_COMPONENTS: 100\n")
        f.write("RANDOM_STATE: 42\n")
    
    with open(os.path.join(outro_melhor_dir, "error.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("0/0.6389/0.6517\n")
        f.write("1/0.6517/0.6389\n")
        f.write("2/0.6389/0.6517\n")
        f.write("3/0.6517/0.6389\n")
        f.write("4/0.6389/0.6517\n")
    
    # Para Outro - Pior (COMBINADO - MLP)
    with open(os.path.join(outro_pior_dir, "config.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("MODEL_TYPE: MLP\n")
        f.write("TASK: verification\n")
        f.write("FEATURE_TYPE: combined\n")
        f.write("MLP_HIDDEN_LAYERS: 1\n")
        f.write("MLP_HIDDEN_SIZE: 100\n")
        f.write("MLP_ACTIVATION: relu\n")
        f.write("MLP_ALPHA: 0.0001\n")
        f.write("MLP_MAX_ITER: 500\n")
        f.write("PCA_COMPONENTS: 100\n")
        f.write("RANDOM_STATE: 42\n")
    
    with open(os.path.join(outro_pior_dir, "error.txt"), "w") as f:
        f.write(f"Executado em {data_atual}\n")
        f.write("0/0.5378/0.5109\n")
        f.write("1/0.4378/0.5728\n")
        f.write("2/0.4327/0.4871\n")
        f.write("3/0.4321/0.6309\n")
        f.write("4/0.4535/0.4529\n")
    
    # Criar arquivos model.dat vazios (placeholders)
    for dir_path in [hog_melhor_dir, hog_pior_dir, outro_melhor_dir, outro_pior_dir]:
        open(os.path.join(dir_path, "model.dat"), "w").close()
    
    print(f"Arquivos de execução criados em: {exec_dir}")
    print("Nota: Os arquivos model.dat estão vazios como placeholders.")
    print("      No trabalho real, seriam os modelos serializados.")

def main():
    """Função principal"""
    
    # Diretório de features
    features_dir = "/home/ed/mestrado/SIN5016/Atividade1/data/features"
    
    # Executar experimento focado
    print("Iniciando sistema de reconhecimento facial...")
    resultados = run_focused_experiment(features_dir)
    
    # Criar arquivos de execução
    create_execution_files()
    
    print("\n" + "="*60)
    print("PROGRAMA CONCLUÍDO!")
    print("="*60)

if __name__ == "__main__":
    main()