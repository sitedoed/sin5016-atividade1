# train_classifiers.py - CÓDIGO OTIMIZADO
import os
import numpy as np
import joblib
import json
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionSystem:
    def __init__(self, features_dir=None):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if features_dir is None:
            self.features_dir = os.path.join(base_dir, 'data', 'features')
        else:
            self.features_dir = os.path.join(base_dir, features_dir.lstrip('./'))
        
        self.results_dir = os.path.join(base_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("="*60)
        print("SISTEMA DE RECONHECIMENTO FACIAL")
        print("="*60)
        print(f"Diretorio de features: {self.features_dir}")
        print("="*60)
        
        print("\nCarregando dados...")
        self.load_data()
    
    def load_data(self):
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Diretorio {self.features_dir} nao existe")
        
        self.X_hog = np.load(os.path.join(self.features_dir, 'hog', 'features.npy'))
        self.X_lbp = np.load(os.path.join(self.features_dir, 'lbp', 'features.npy'))
        self.X_combined = np.load(os.path.join(self.features_dir, 'combined', 'features.npy'))
        
        self.y_ids = np.load(os.path.join(self.features_dir, 'metadata', 'labels.npy'))
        
        self.pairs = np.load(os.path.join(self.features_dir, 'verification_pairs', 'pairs.npy'))
        self.y_verif = np.load(os.path.join(self.features_dir, 'verification_pairs', 'pair_labels.npy'))
        
        mapping_path = os.path.join(self.features_dir, 'metadata', 'label_to_person.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.label_to_person = json.load(f)
        else:
            self.label_to_person = {}
        
        print(f"Dados carregados:")
        print(f"  Identificacao: {len(self.y_ids)} imagens, {len(np.unique(self.y_ids))} classes")
        print(f"  Verificacao: {len(self.y_verif)} pares ({sum(self.y_verif)} positivos)")
    
    def prepare_identification_data(self, feature_type='hog', test_size=0.2, val_size=0.1, 
                                  random_state=42, pca_components=50, max_samples=10000):
        """Prepara dados para identificacao com limite de amostras"""
        if feature_type == 'hog':
            X = self.X_hog
        elif feature_type == 'lbp':
            X = self.X_lbp
        else:
            X = self.X_combined
        
        y = self.y_ids
        
        # Limitar numero de amostras para acelerar
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, random_state=random_state, stratify=y_temp
        )
        
        if pca_components and X_train.shape[1] > pca_components:
            print(f"  PCA: {X_train.shape[1]} -> {pca_components} componentes")
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
        
        print(f"\nDados IDENTIFICACAO ({feature_type}):")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Validacao: {X_val.shape[0]} amostras")
        print(f"  Teste: {X_test.shape[0]} amostras")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"  Dimensoes: {X_train.shape[1]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_verification_data(self, feature_type='hog', test_size=0.2, 
                                 random_state=42, max_samples=5000, pca_components=100):
        """Prepara dados para verificacao"""
        if feature_type == 'hog':
            features = self.X_hog
        elif feature_type == 'lbp':
            features = self.X_lbp
        else:
            features = self.X_combined
        
        import random
        if max_samples and len(self.pairs) > max_samples:
            indices = random.sample(range(len(self.pairs)), max_samples)
            pairs_sample = [self.pairs[i] for i in indices]
            y_sample = [self.y_verif[i] for i in indices]
        else:
            pairs_sample = self.pairs
            y_sample = self.y_verif
        
        X_pairs = []
        for i, j in pairs_sample:
            pair_feature = np.concatenate([features[i], features[j]])
            X_pairs.append(pair_feature)
        
        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_sample)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_pairs, y_pairs, test_size=test_size, random_state=random_state, stratify=y_pairs
        )
        
        if pca_components and X_train.shape[1] > pca_components:
            print(f"  PCA: {X_train.shape[1]} -> {pca_components} componentes")
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        
        print(f"\nDados VERIFICACAO ({feature_type}):")
        print(f"  Treino: {X_train.shape[0]} pares")
        print(f"  Teste: {X_test.shape[0]} pares")
        print(f"  Positivos: {sum(y_train)}/{sum(y_test)}")
        print(f"  Dimensao: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_mlp_with_epochs(self, X_train, y_train, X_val, y_val, task_type='identification',
                             hidden_layers=(100,), max_epochs=200, learning_rate=0.001,
                             patience=20, min_delta=0.001):
        print(f"\nTreinando MLP para {task_type}...")
        print(f"Arquitetura: {hidden_layers}")
        print(f"Max epocas: {max_epochs}, Patience: {patience}")
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1,
            learning_rate_init=learning_rate,
            warm_start=True,
            random_state=42,
            early_stopping=False
        )
        
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'best_epoch': 0
        }
        
        best_val_loss = float('inf')
        best_model_params = None
        patience_counter = 0
        
        for epoch in range(max_epochs):
            model.fit(X_train, y_train)
            
            train_loss = model.loss_
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            
            val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_loss = 1.0 - val_acc
            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['train_acc'].append(float(train_acc))
            history['val_acc'].append(float(val_acc))
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_params = {
                    'coefs_': [c.copy() for c in model.coefs_],
                    'intercepts_': [i.copy() for i in model.intercepts_],
                    'n_iter_': model.n_iter_,
                    'loss_': model.loss_
                }
                history['best_epoch'] = epoch + 1
                patience_counter = 0
                if epoch % 10 == 0:
                    print(f"  Epoca {epoch+1:3d} - Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} *")
            else:
                patience_counter += 1
                if epoch % 20 == 0:
                    print(f"  Epoca {epoch+1:3d} - Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping epoca {epoch+1}")
                print(f"Melhor epoca: {history['best_epoch']} (val_acc: {history['val_acc'][history['best_epoch']-1]:.4f})")
                break
        
        if best_model_params is not None:
            model.coefs_ = best_model_params['coefs_']
            model.intercepts_ = best_model_params['intercepts_']
            model.n_iter_ = best_model_params['n_iter_']
            model.loss_ = best_model_params['loss_']
        
        print(f"Treinamento finalizado:")
        print(f"  Total epocas: {len(history['epoch'])}")
        print(f"  Melhor epoca: {history['best_epoch']}")
        print(f"  Melhor val_acc: {history['val_acc'][history['best_epoch']-1]:.4f}")
        
        return model, history
    
    def train_svm(self, X_train, y_train, X_val, y_val, task_type='identification',
                  C=1.0, kernel='linear', max_iter=1000):
        print(f"\nTreinando SVM para {task_type}...")
        
        # Para identificacao com muitos dados, usar LinearSVC (mais rapido)
        if task_type == 'identification':
            print("  Usando LinearSVC (otimizado para multi-classe)...")
            model = LinearSVC(
                C=C,
                random_state=42,
                max_iter=max_iter,
                verbose=1,
                dual=False  # Mais rapido quando n_samples > n_features
            )
        else:
            # Para verificacao (binario), usar SVC tradicional
            if X_train.shape[0] > 5000:
                kernel = 'linear'
                max_iter = 500
            
            model = SVC(
                C=C,
                kernel=kernel,
                probability=True,
                random_state=42,
                max_iter=max_iter,
                verbose=False
            )
        
        print(f"  Iniciando treinamento...")
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        print(f"SVM Treinado:")
        print(f"  Tipo: {model.__class__.__name__}")
        print(f"  Acuraria validacao: {val_acc:.4f}")
        
        return model
    
    def cross_validation_mlp(self, X, y, feature_type='hog', task_type='identification', 
                            n_folds=5, hidden_layers=(100,), max_epochs=200):
        print(f"\n{task_type.upper()} - {feature_type.upper()} - MLP")
        print(f"5-fold cross validation...")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scores = []
        histories = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/5:")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model, history = self.train_mlp_with_epochs(
                X_train, y_train, X_val, y_val, task_type,
                hidden_layers=hidden_layers, max_epochs=max_epochs
            )
            
            score = accuracy_score(y_val, model.predict(X_val))
            scores.append(score)
            histories.append(history)
            models.append(model)
            
            print(f"  Fold {fold} - Val Acc: {score:.4f}")
        
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores,
            'models': models,
            'histories': histories,
            'feature_type': feature_type,
            'task_type': task_type,
            'model_type': 'mlp',
            'n_folds': n_folds
        }
        
        print(f"\nResultados 5-fold CV:")
        print(f"  Media: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def cross_validation_svm(self, X, y, feature_type='hog', task_type='identification', n_folds=5):
        print(f"\n{task_type.upper()} - {feature_type.upper()} - SVM")
        print(f"5-fold cross validation...")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/5:")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.train_svm(X_train, y_train, X_val, y_val, task_type)
            
            score = accuracy_score(y_val, model.predict(X_val))
            scores.append(score)
            models.append(model)
            
            print(f"  Fold {fold} - Val Acc: {score:.4f}")
        
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores,
            'models': models,
            'feature_type': feature_type,
            'task_type': task_type,
            'model_type': 'svm',
            'n_folds': n_folds
        }
        
        print(f"\nResultados 5-fold CV:")
        print(f"  Media: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def evaluate_model(self, model, X_test, y_test, feature_type, task_type, model_type, history=None):
        print(f"\nAvaliacao final - {task_type} - {feature_type} - {model_type}:")
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if task_type == 'identification':
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"Acuraria: {acc:.4f}")
            print(f"Precisao media: {report['macro avg']['precision']:.4f}")
            print(f"Revocacao media: {report['macro avg']['recall']:.4f}")
            
            self.plot_confusion_matrix(cm, feature_type, task_type, model_type)
            
        else:
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Para LinearSVC nao tem predict_proba
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob[:, 1])
                print(f"AUC: {auc:.4f}")
            
            print(f"Acuraria: {acc:.4f}")
            print(f"Precisao: {report['1']['precision']:.4f}")
            print(f"Revocacao: {report['1']['recall']:.4f}")
            
            self.plot_confusion_matrix(cm, feature_type, task_type, model_type)
        
        results = {
            'accuracy': float(acc),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_type': feature_type,
            'task_type': task_type,
            'model_type': model_type
        }
        
        if history and model_type == 'mlp':
            results['training_history'] = {
                'epochs': history['epoch'],
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'train_acc': history['train_acc'],
                'val_acc': history['val_acc'],
                'best_epoch': history['best_epoch']
            }
            
            self.plot_training_history(history, feature_type, task_type)
        
        return results
    
    def plot_confusion_matrix(self, cm, feature_type, task_type, model_type):
        plt.figure(figsize=(10, 8))
        
        if task_type == 'identification':
            sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de Confusao (20 classes)\n{task_type} - {feature_type} - {model_type}')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Diferente', 'Mesma'],
                       yticklabels=['Diferente', 'Mesma'])
            plt.title(f'Matriz de Confusao\n{task_type} - {feature_type} - {model_type}')
        
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        
        plot_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        filename = f'cm_{task_type}_{feature_type}_{model_type}.png'
        plt.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close()
    
    def plot_training_history(self, history, feature_type, task_type):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = history['epoch']
        
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Treino')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validacao')
        axes[0, 0].axvline(x=history['best_epoch'], color='g', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Epoca')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss durante treinamento')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Treino')
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validacao')
        axes[0, 1].axvline(x=history['best_epoch'], color='g', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoca')
        axes[0, 1].set_ylabel('Acuraria')
        axes[0, 1].set_title('Acuraria durante treinamento')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].semilogy(epochs, history['train_loss'], 'b-')
        axes[1, 0].semilogy(epochs, history['val_loss'], 'r-')
        axes[1, 0].axvline(x=history['best_epoch'], color='g', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoca')
        axes[1, 0].set_ylabel('Loss (log)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['train_acc'], history['val_acc'], 'o-', alpha=0.5)
        axes[1, 1].set_xlabel('Acuraria Treino')
        axes[1, 1].set_ylabel('Acuraria Validacao')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Historico - {task_type} - {feature_type}', fontsize=14)
        plt.tight_layout()
        
        plot_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        filename = f'history_{task_type}_{feature_type}_mlp.png'
        plt.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close()
    
    def save_model_and_results(self, model, results, feature_type, task_type, model_type, history=None):
        model_dir = os.path.join(self.results_dir, 'models', task_type, feature_type)
        log_dir = os.path.join(self.results_dir, 'logs', task_type, feature_type)
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        basename = f'{model_type}_{timestamp}'
        
        model_path = os.path.join(model_dir, f'{basename}.pkl')
        joblib.dump(model, model_path)
        
        results_path = os.path.join(log_dir, f'{basename}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if model_type == 'mlp':
            config = {
                'hidden_layers': model.hidden_layer_sizes,
                'activation': model.activation,
                'solver': model.solver,
                'learning_rate': model.learning_rate,
                'max_iter': model.max_iter
            }
            
            if history:
                history_path = os.path.join(log_dir, f'{basename}_history.json')
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
        else:
            config = {
                'model_type': model.__class__.__name__,
                'C': getattr(model, 'C', 'N/A'),
                'kernel': getattr(model, 'kernel', 'linear'),
                'max_iter': getattr(model, 'max_iter', 'N/A')
            }
        
        config_path = os.path.join(log_dir, f'{basename}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Modelo e resultados salvos:")
        print(f"  Modelo: {model_path}")
        
        return basename
    
    def run_focused_experiment(self):
        """Executa experimento focado nos melhores cenarios"""
        print("\n" + "="*60)
        print("EXPERIMENTO FOCADO - 5-FOLD CV")
        print("Testando cenarios mais promissores:")
        print("  1. VERIFICACAO: HOG e COMBINADO")
        print("  2. IDENTIFICACAO: HOG (limitado a 5k amostras)")
        print("="*60)
        
        # Configuracoes otimizadas
        verification_features = ['hog', 'combined']
        identification_features = ['hog']  # Apenas HOG para identificacao
        
        all_results = {}
        
        # 1. TESTAR VERIFICACAO
        print("\n" + "="*50)
        print("FASE 1: VERIFICACAO FACIAL")
        print("="*50)
        
        all_results['verification'] = {}
        
        for feature in verification_features:
            print(f"\n{'='*40}")
            print(f"VERIFICACAO - {feature.upper()}")
            print('='*40)
            
            try:
                X_train, X_test, y_train, y_test = self.prepare_verification_data(
                    feature_type=feature, max_samples=3000, pca_components=100
                )
                X = X_train
                y = y_train
                
                # MLP
                print(f"\n--- MLP ---")
                mlp_cv = self.cross_validation_mlp(
                    X, y, feature, 'verification', n_folds=5,
                    hidden_layers=(100,), max_epochs=150
                )
                
                best_mlp_idx = np.argmax(mlp_cv['scores'])
                best_mlp = mlp_cv['models'][best_mlp_idx]
                best_history = mlp_cv['histories'][best_mlp_idx]
                
                mlp_results = self.evaluate_model(best_mlp, X_test, y_test, feature, 
                                                 'verification', 'mlp', best_history)
                mlp_basename = self.save_model_and_results(best_mlp, mlp_results, 
                                                          feature, 'verification', 'mlp', best_history)
                
                # SVM
                print(f"\n--- SVM ---")
                svm_cv = self.cross_validation_svm(X, y, feature, 'verification', n_folds=5)
                
                best_svm_idx = np.argmax(svm_cv['scores'])
                best_svm = svm_cv['models'][best_svm_idx]
                
                svm_results = self.evaluate_model(best_svm, X_test, y_test, feature, 
                                                 'verification', 'svm')
                svm_basename = self.save_model_and_results(best_svm, svm_results, 
                                                          feature, 'verification', 'svm')
                
                all_results['verification'][feature] = {
                    'mlp': {
                        'cv_mean': float(mlp_cv['mean_score']),
                        'cv_std': float(mlp_cv['std_score']),
                        'test_accuracy': mlp_results['accuracy'],
                        'best_epoch': best_history['best_epoch'],
                        'model_file': mlp_basename
                    },
                    'svm': {
                        'cv_mean': float(svm_cv['mean_score']),
                        'cv_std': float(svm_cv['std_score']),
                        'test_accuracy': svm_results['accuracy'],
                        'model_file': svm_basename
                    }
                }
                
            except Exception as e:
                print(f"Erro ao processar verificacao-{feature}: {e}")
                continue
        
        # 2. TESTAR IDENTIFICACAO (limitado)
        print("\n" + "="*50)
        print("FASE 2: IDENTIFICACAO (LIMITADO)")
        print("="*50)
        
        all_results['identification'] = {}
        
        for feature in identification_features:
            print(f"\n{'='*40}")
            print(f"IDENTIFICACAO - {feature.upper()} (5k amostras)")
            print('='*40)
            
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_identification_data(
                    feature_type=feature, pca_components=50, max_samples=5000
                )
                X = np.vstack([X_train, X_val])
                y = np.concatenate([y_train, y_val])
                
                # MLP
                print(f"\n--- MLP ---")
                mlp_cv = self.cross_validation_mlp(
                    X, y, feature, 'identification', n_folds=5,
                    hidden_layers=(100, 50), max_epochs=200
                )
                
                best_mlp_idx = np.argmax(mlp_cv['scores'])
                best_mlp = mlp_cv['models'][best_mlp_idx]
                best_history = mlp_cv['histories'][best_mlp_idx]
                
                mlp_results = self.evaluate_model(best_mlp, X_test, y_test, feature, 
                                                 'identification', 'mlp', best_history)
                mlp_basename = self.save_model_and_results(best_mlp, mlp_results, 
                                                          feature, 'identification', 'mlp', best_history)
                
                # SVM (LinearSVC otimizado)
                print(f"\n--- SVM (LinearSVC) ---")
                svm_cv = self.cross_validation_svm(X, y, feature, 'identification', n_folds=5)
                
                best_svm_idx = np.argmax(svm_cv['scores'])
                best_svm = svm_cv['models'][best_svm_idx]
                
                svm_results = self.evaluate_model(best_svm, X_test, y_test, feature, 
                                                 'identification', 'svm')
                svm_basename = self.save_model_and_results(best_svm, svm_results, 
                                                          feature, 'identification', 'svm')
                
                all_results['identification'][feature] = {
                    'mlp': {
                        'cv_mean': float(mlp_cv['mean_score']),
                        'cv_std': float(mlp_cv['std_score']),
                        'test_accuracy': mlp_results['accuracy'],
                        'best_epoch': best_history['best_epoch'],
                        'model_file': mlp_basename
                    },
                    'svm': {
                        'cv_mean': float(svm_cv['mean_score']),
                        'cv_std': float(svm_cv['std_score']),
                        'test_accuracy': svm_results['accuracy'],
                        'model_file': svm_basename
                    }
                }
                
            except Exception as e:
                print(f"Erro ao processar identificacao-{feature}: {e}")
                continue
        
        # Salvar resumo
        summary_path = os.path.join(self.results_dir, 'experiment_summary_focused.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EXPERIMENTO FOCADO CONCLUIDO!")
        print(f"{'='*60}")
        
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results):
        print("\nRESUMO DOS RESULTADOS (5-fold CV):")
        print("="*60)
        
        for task in ['identification', 'verification']:
            if task in results:
                print(f"\n{task.upper()}:")
                print("-"*40)
                
                for feature in results[task]:
                    task_data = results[task][feature]
                    print(f"\n  {feature.upper()}:")
                    
                    for model_type in ['mlp', 'svm']:
                        if model_type in task_data:
                            model_data = task_data[model_type]
                            print(f"    {model_type.upper()}:")
                            print(f"      CV (5-fold): {model_data['cv_mean']:.4f} (±{model_data['cv_std']:.4f})")
                            print(f"      Teste: {model_data['test_accuracy']:.4f}")
                            if model_type == 'mlp' and 'best_epoch' in model_data:
                                print(f"      Melhor epoca: {model_data['best_epoch']}")

def main():
    print("SISTEMA DE RECONHECIMENTO FACIAL - ATIVIDADE 1")
    print("Executando experimento focado (mais rapido)")
    
    system = FaceRecognitionSystem()
    
    print("\nExecutando experimento focado...")
    print("Isso testara cenarios mais promissores de forma eficiente.")
    
    results = system.run_focused_experiment()
    
    print(f"\nResultados salvos em: {system.results_dir}")

if __name__ == "__main__":
    main()