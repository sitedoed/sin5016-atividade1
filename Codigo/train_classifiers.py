# train_classifiers.py
import os
import numpy as np
import joblib
import json
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    def __init__(self, features_dir='../data/features'):
        self.features_dir = features_dir
        
        print("="*60)
        print("SISTEMA DE RECONHECIMENTO FACIAL")
        print("="*60)
        
        print("\nCarregando dados...")
        self.load_data()
        
        self.results_dir = '../results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self):
        self.X_hog = np.load(os.path.join(self.features_dir, 'hog', 'features.npy'))
        self.X_lbp = np.load(os.path.join(self.features_dir, 'lbp', 'features.npy'))
        self.X_combined = np.load(os.path.join(self.features_dir, 'combined', 'features.npy'))
        
        self.y_ids = np.load(os.path.join(self.features_dir, 'metadata', 'labels.npy'))
        
        self.pairs = np.load(os.path.join(self.features_dir, 'verification_pairs', 'pairs.npy'))
        self.y_verif = np.load(os.path.join(self.features_dir, 'verification_pairs', 'pair_labels.npy'))
        
        with open(os.path.join(self.features_dir, 'metadata', 'label_to_person.json'), 'r') as f:
            self.label_to_person = json.load(f)
        
        print(f"Dados carregados:")
        print(f"  Identificacao: {len(self.y_ids)} imagens, {len(np.unique(self.y_ids))} pessoas")
        print(f"  Verificacao: {len(self.y_verif)} pares ({sum(self.y_verif)} positivos)")
        print(f"  Dimensoes - HOG: {self.X_hog.shape}, LBP: {self.X_lbp.shape}")
    
    def prepare_identification_data(self, feature_type='hog', test_size=0.3, val_size=0.1, 
                                  random_state=42, pca_components=100):
        if feature_type == 'hog':
            X = self.X_hog
        elif feature_type == 'lbp':
            X = self.X_lbp
        else:
            X = self.X_combined
        
        y = self.y_ids
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, random_state=random_state, stratify=y_temp
        )
        
        if pca_components and X_train.shape[1] > pca_components:
            print(f"  Aplicando PCA: {X_train.shape[1]} -> {pca_components} componentes")
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
        
        print(f"\nDados para IDENTIFICACAO ({feature_type}):")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Validacao: {X_val.shape[0]} amostras")
        print(f"  Teste: {X_test.shape[0]} amostras")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"  Dimensoes: {X_train.shape[1]}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_type': feature_type
        }
    
    def prepare_verification_data(self, feature_type='hog', test_size=0.3, random_state=42, 
                                 max_samples=2000, pca_components=100):
        if feature_type == 'hog':
            features = self.X_hog
        elif feature_type == 'lbp':
            features = self.X_lbp
        else:
            features = self.X_combined
        
        import random
        if len(self.pairs) > max_samples:
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
            print(f"  Aplicando PCA: {X_train.shape[1]} -> {pca_components} componentes")
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        
        print(f"\nDados para VERIFICACAO ({feature_type}):")
        print(f"  Treino: {X_train.shape[0]} pares")
        print(f"  Teste: {X_test.shape[0]} pares")
        print(f"  Positivos (treino): {sum(y_train)}")
        print(f"  Negativos (treino): {len(y_train) - sum(y_train)}")
        print(f"  Dimensao por par: {X_train.shape[1]}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'feature_type': feature_type
        }
    
    def train_mlp_with_epochs(self, X_train, y_train, X_val, y_val, task_type='identification',
                             hidden_layers=(100,), max_epochs=500, learning_rate=0.001,
                             patience=20, min_delta=0.001, batch_size=32):
        print(f"\nTreinando MLP para {task_type}...")
        print(f"Arquitetura: {hidden_layers}")
        print(f"Max epocas: {max_epochs}, Patience: {patience}, Batch size: {batch_size}")
        
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
        
        print("\nIniciando treinamento...")
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
                if epoch % 5 == 0:
                    print(f"  Epoca {epoch+1:3d} - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} *")
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    print(f"  Epoca {epoch+1:3d} - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping na epoca {epoch+1}")
                print(f"Melhor epoca: {history['best_epoch']} com val_loss: {best_val_loss:.4f}")
                break
        
        if best_model_params is not None:
            model.coefs_ = best_model_params['coefs_']
            model.intercepts_ = best_model_params['intercepts_']
            model.n_iter_ = best_model_params['n_iter_']
            model.loss_ = best_model_params['loss_']
        
        print(f"\nTreinamento finalizado:")
        print(f"  Total de epocas: {len(history['epoch'])}")
        print(f"  Melhor epoca: {history['best_epoch']}")
        print(f"  Melhor val_loss: {best_val_loss:.4f}")
        print(f"  Melhor val_acc: {history['val_acc'][history['best_epoch']-1]:.4f}")
        
        return model, history
    
    def train_svm(self, X_train, y_train, X_val, y_val, task_type='identification',
                  C=1.0, kernel='linear', gamma='scale', max_iter=1000):
        print(f"\nTreinando SVM para {task_type}...")
        
        if X_train.shape[0] > 5000 or X_train.shape[1] > 1000:
            print(f"  Dados grandes ({X_train.shape[0]} amostras, {X_train.shape[1]} features)")
            print(f"  Usando kernel linear e max_iter reduzido")
            kernel = 'linear'
            max_iter = 500
        
        try:
            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                random_state=42,
                max_iter=max_iter,
                verbose=False
            )
            
            if task_type == 'identification':
                model.decision_function_shape = 'ovr'
            
            print(f"  Iniciando treinamento SVM...")
            model.fit(X_train, y_train)
            
        except Exception as e:
            print(f"  Erro no SVC: {e}")
            print("  Usando SGDClassifier (SVM com gradiente descendente)...")
            
            model = SGDClassifier(
                loss='hinge',
                alpha=1/(C * X_train.shape[0]),
                max_iter=1000,
                random_state=42,
                verbose=0
            )
            model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        print(f"SVM Treinado:")
        print(f"  Kernel: {kernel}, C: {C}")
        print(f"  Acuraria treinamento: {train_acc:.4f}")
        print(f"  Acuraria validacao: {val_acc:.4f}")
        
        return model
    
    def cross_validation_mlp(self, X, y, feature_type='hog', task_type='identification', 
                            n_folds=3, hidden_layers=(50,), max_epochs=50):
        print(f"\n{task_type.upper()} - {feature_type.upper()} - MLP")
        print(f"Executando {n_folds}-fold cross validation...")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scores = []
        histories = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{n_folds}:")
            
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
            
            print(f"  Fold {fold} - Melhor val_acc: {history['val_acc'][history['best_epoch']-1]:.4f}")
            print(f"  Fold {fold} - Acuraria final: {score:.4f}")
        
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
        
        print(f"\nResultados {n_folds}-fold CV:")
        print(f"  Media: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        print(f"  Scores individuais: {[f'{s:.4f}' for s in scores]}")
        
        return results
    
    def cross_validation_svm(self, X, y, feature_type='hog', task_type='identification', n_folds=3):
        print(f"\n{task_type.upper()} - {feature_type.upper()} - SVM")
        print(f"Executando {n_folds}-fold cross validation...")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{n_folds}:")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.train_svm(X_train, y_train, X_val, y_val, task_type)
            
            score = accuracy_score(y_val, model.predict(X_val))
            scores.append(score)
            models.append(model)
            
            print(f"  Fold {fold} - Acuraria: {score:.4f}")
        
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
        
        print(f"\nResultados {n_folds}-fold CV:")
        print(f"  Media: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        print(f"  Scores individuais: {[f'{s:.4f}' for s in scores]}")
        
        return results
    
    def evaluate_model(self, model, X_test, y_test, feature_type, task_type, model_type, history=None):
        print(f"\nAvaliacao final - {task_type} - {feature_type} - {model_type}:")
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        
        if task_type == 'identification':
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"Acuraria: {acc:.4f}")
            print(f"Acuraria media por classe: {report['macro avg']['precision']:.4f}")
            print(f"Revocacao media: {report['macro avg']['recall']:.4f}")
            
            self.plot_confusion_matrix(cm, feature_type, task_type, model_type)
            
        else:
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            if y_prob is not None:
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
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
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
            plt.title(f'Matriz de Confusao (Primeiras 20 classes)\n{feature_type} - {model_type}')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Diferente', 'Mesma'],
                       yticklabels=['Diferente', 'Mesma'])
            plt.title(f'Matriz de Confusao\n{feature_type} - {model_type}')
        
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
        axes[0, 0].set_title('Loss durante o treinamento')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Treino')
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validacao')
        axes[0, 1].axvline(x=history['best_epoch'], color='g', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoca')
        axes[0, 1].set_ylabel('Acuraria')
        axes[0, 1].set_title('Acuraria durante o treinamento')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].semilogy(epochs, history['train_loss'], 'b-')
        axes[1, 0].semilogy(epochs, history['val_loss'], 'r-')
        axes[1, 0].axvline(x=history['best_epoch'], color='g', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoca')
        axes[1, 0].set_ylabel('Loss (log)')
        axes[1, 0].set_title('Loss (escala logaritmica)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['train_acc'], history['val_acc'], 'o-', alpha=0.5)
        axes[1, 1].set_xlabel('Acuraria Treino')
        axes[1, 1].set_ylabel('Acuraria Validacao')
        axes[1, 1].set_title('Treino vs Validacao')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Historico de Treinamento - {task_type} - {feature_type}', fontsize=14)
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
            if hasattr(model, 'kernel'):
                config = {
                    'C': model.C,
                    'kernel': model.kernel,
                    'gamma': model.gamma,
                    'decision_function_shape': getattr(model, 'decision_function_shape', 'ovr')
                }
            else:
                config = {
                    'model_type': 'SGDClassifier',
                    'loss': model.loss,
                    'alpha': model.alpha
                }
        
        config_path = os.path.join(log_dir, f'{basename}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nModelo e resultados salvos:")
        print(f"  Modelo: {model_path}")
        print(f"  Resultados: {results_path}")
        print(f"  Configuracao: {config_path}")
        
        return basename
    
    def run_experiment_simple(self):
        print("\n" + "="*60)
        print("EXPERIMENTO SIMPLIFICADO (TESTE RAPIDO)")
        print("="*60)
        
        feature_types = ['hog']
        task_types = ['verification']
        
        all_results = {}
        
        for task in task_types:
            all_results[task] = {}
            
            for feature in feature_types:
                print(f"\n{'='*40}")
                print(f"TAREFA: {task.upper()} | CARACTERISTICA: {feature.upper()}")
                print('='*40)
                
                data = self.prepare_verification_data(feature_type=feature, max_samples=2000, pca_components=50)
                X = data['X_train']
                y = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']
                
                print(f"\n--- MLP ---")
                mlp_cv = self.cross_validation_mlp(X, y, feature, task, n_folds=3, 
                                                  hidden_layers=(50,), max_epochs=30)
                
                best_mlp_idx = np.argmax(mlp_cv['scores'])
                best_mlp = mlp_cv['models'][best_mlp_idx]
                best_history = mlp_cv['histories'][best_mlp_idx]
                
                mlp_results = self.evaluate_model(best_mlp, X_test, y_test, feature, 
                                                 task, 'mlp', best_history)
                mlp_basename = self.save_model_and_results(best_mlp, mlp_results, 
                                                          feature, task, 'mlp', best_history)
                
                print(f"\n--- SVM ---")
                svm_cv = self.cross_validation_svm(X, y, feature, task, n_folds=3)
                
                best_svm_idx = np.argmax(svm_cv['scores'])
                best_svm = svm_cv['models'][best_svm_idx]
                
                svm_results = self.evaluate_model(best_svm, X_test, y_test, feature, 
                                                 task, 'svm')
                svm_basename = self.save_model_and_results(best_svm, svm_results, 
                                                          feature, task, 'svm')
                
                all_results[task][feature] = {
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
        
        summary_path = os.path.join(self.results_dir, 'experiment_summary_simple.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EXPERIMENTO SIMPLIFICADO CONCLUIDO!")
        print(f"{'='*60}")
        
        self.print_summary(all_results)
        
        return all_results
    
    def run_full_experiment(self):
        print("\n" + "="*60)
        print("EXPERIMENTO COMPLETO (PODE DEMORAR HORAS - OTIMIZADO)")
        print("="*60)
        
        feature_types = ['hog', 'lbp', 'combined']
        task_types = ['verification']  # Apenas verificacao para teste
        
        all_results = {}
        
        for task in task_types:
            all_results[task] = {}
            
            for feature in feature_types:
                print(f"\n{'='*40}")
                print(f"TAREFA: {task.upper()} | CARACTERISTICA: {feature.upper()}")
                print('='*40)
                
                if task == 'identification':
                    data = self.prepare_identification_data(feature_type=feature, pca_components=100)
                    X = data['X_train']
                    y = data['y_train']
                    X_test = data['X_test']
                    y_test = data['y_test']
                else:
                    data = self.prepare_verification_data(feature_type=feature, max_samples=5000, pca_components=100)
                    X = data['X_train']
                    y = data['y_train']
                    X_test = data['X_test']
                    y_test = data['y_test']
                
                print(f"\n--- MLP ---")
                mlp_cv = self.cross_validation_mlp(X, y, feature, task, n_folds=3, 
                                                  hidden_layers=(100,), max_epochs=100)
                
                best_mlp_idx = np.argmax(mlp_cv['scores'])
                best_mlp = mlp_cv['models'][best_mlp_idx]
                best_history = mlp_cv['histories'][best_mlp_idx]
                
                mlp_results = self.evaluate_model(best_mlp, X_test, y_test, feature, 
                                                 task, 'mlp', best_history)
                mlp_basename = self.save_model_and_results(best_mlp, mlp_results, 
                                                          feature, task, 'mlp', best_history)
                
                print(f"\n--- SVM ---")
                svm_cv = self.cross_validation_svm(X, y, feature, task, n_folds=3)
                
                best_svm_idx = np.argmax(svm_cv['scores'])
                best_svm = svm_cv['models'][best_svm_idx]
                
                svm_results = self.evaluate_model(best_svm, X_test, y_test, feature, 
                                                 task, 'svm')
                svm_basename = self.save_model_and_results(best_svm, svm_results, 
                                                          feature, task, 'svm')
                
                all_results[task][feature] = {
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
        
        summary_path = os.path.join(self.results_dir, 'experiment_summary_full.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EXPERIMENTO COMPLETO CONCLUIDO!")
        print(f"{'='*60}")
        
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results):
        print("\nRESUMO DOS RESULTADOS:")
        print("="*60)
        
        for task in results.keys():
            print(f"\n{task.upper()}:")
            print("-"*40)
            
            for feature in results[task].keys():
                task_data = results[task][feature]
                print(f"\n  {feature.upper()}:")
                
                for model_type in ['mlp', 'svm']:
                    model_data = task_data[model_type]
                    print(f"    {model_type.upper()}:")
                    print(f"      CV: {model_data['cv_mean']:.4f} (±{model_data['cv_std']:.4f})")
                    print(f"      Teste: {model_data['test_accuracy']:.4f}")
                    if model_type == 'mlp':
                        print(f"      Melhor epoca: {model_data.get('best_epoch', 'N/A')}")

def main():
    system = FaceRecognitionSystem()
    
    print("\nEscolha o tipo de experimento:")
    print("1. Simplificado (teste rapido - 2,000 pares, PCA 50, 30 epocas)")
    print("2. Completo otimizado (5,000 pares, PCA 100, 100 epocas)")
    
    choice = input("\nDigite 1 ou 2: ").strip()
    
    if choice == '1':
        print("\nExecutando experimento simplificado...")
        results = system.run_experiment_simple()
    elif choice == '2':
        print("\nExecutando experimento completo otimizado...")
        results = system.run_full_experiment()
    else:
        print("Opcao invalida. Executando experimento simplificado por padrao.")
        results = system.run_experiment_simple()
    
    print(f"\nTodos os resultados salvos em: {system.results_dir}")

if __name__ == "__main__":
    main()