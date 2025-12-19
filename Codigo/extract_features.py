# Codigo/extract_features_final.py
import os
import cv2
import numpy as np
import joblib
import argparse
from tqdm import tqdm
import yaml
from skimage import feature
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import Counter
import json

class FeatureExtractorWithLabels:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.scaler_hog = StandardScaler()
        self.scaler_lbp = StandardScaler()
        self.scaler_combined = StandardScaler()
        
    def _get_default_config(self):
        return {
            'hog': {
                'orientations': 9,
                'pixels_per_cell': (8, 8),
                'cells_per_block': (2, 2),
                'block_norm': 'L2-Hys',
                'transform_sqrt': True,
                'feature_vector': True
            },
            'lbp': {
                'num_points': 24,
                'radius': 8,
                'method': 'uniform'
            },
            'general': {
                'image_size': (64, 64),
                'normalize_features': True,
                'save_scalers': True,
                'min_images_per_person': 10,  # Minimo para validação cruzada
                'max_persons': None,  # None para processar todas
                'image_dir': '../Images/Selected_images',
                'identity_file': '../data/raw/identity_CelebA.txt'
            }
        }
    
    def load_identity_mapping(self, identity_file, image_dir):
        """
        Carrega mapeamento e filtra apenas imagens existentes
        
        Args:
            identity_file: Caminho para identity_CelebA.txt
            image_dir: Diretorio com imagens selecionadas
            
        Returns:
            dict: Mapeamento filename -> person_id
        """
        print(f"Carregando mapeamento de {identity_file}...")
        
        # Carregar arquivo de identidade
        df = pd.read_csv(identity_file, sep=' ', header=None, 
                        names=['image_id', 'person_id'])
        
        # Listar imagens disponíveis
        available_images = set(os.listdir(image_dir))
        print(f"Imagens disponíveis: {len(available_images)}")
        
        # Filtrar apenas imagens que temos
        df_filtered = df[df['image_id'].isin(available_images)]
        
        # Criar dicionario
        mapping = dict(zip(df_filtered['image_id'], df_filtered['person_id']))
        
        # Estatisticas
        counts = Counter(mapping.values())
        print(f"\nESTATISTICAS DO MAPEAMENTO:")
        print(f"  Imagens mapeadas: {len(mapping)}")
        print(f"  Pessoas únicas: {len(counts)}")
        print(f"  Imagens por pessoa (média): {np.mean(list(counts.values())):.1f}")
        print(f"  Imagens por pessoa (mín): {min(counts.values())}")
        print(f"  Imagens por pessoa (máx): {max(counts.values())}")
        
        # Top 10 pessoas com mais imagens
        print(f"\nTop 10 pessoas com mais imagens:")
        for person_id, count in counts.most_common(10):
            print(f"  Pessoa {person_id}: {count} imagens")
        
        return mapping, counts
    
    def filter_persons(self, mapping, counts, min_images=10, max_persons=None):
        """
        Filtra pessoas com poucas imagens
        
        Args:
            mapping: Dicionario filename -> person_id
            counts: Counter com contagem por pessoa
            min_images: Minimo de imagens por pessoa
            max_persons: Maximo de pessoas (None para todas)
            
        Returns:
            dict: Mapeamento filtrado
        """
        # Selecionar pessoas com pelo menos min_images
        valid_persons = {person_id for person_id, count in counts.items() 
                        if count >= min_images}
        
        # Limitar numero de pessoas se necessario
        if max_persons and len(valid_persons) > max_persons:
            # Selecionar pessoas com mais imagens
            valid_persons = set([person_id for person_id, _ in 
                               counts.most_common(max_persons)])
        
        # Filtrar mapeamento
        filtered_mapping = {filename: person_id 
                          for filename, person_id in mapping.items()
                          if person_id in valid_persons}
        
        # Recalcular contagens
        filtered_counts = Counter(filtered_mapping.values())
        
        print(f"\nFILTRO APLICADO:")
        print(f"  Pessoas originais: {len(counts)}")
        print(f"  Pessoas válidas (≥{min_images} imagens): {len(valid_persons)}")
        print(f"  Imagens válidas: {len(filtered_mapping)}")
        print(f"  Imagens por pessoa (nova média): {np.mean(list(filtered_counts.values())):.1f}")
        
        return filtered_mapping, filtered_counts
    
    def extract_hog(self, image):
        """Extrai características HOG"""
        if image.shape != self.config['general']['image_size']:
            image = cv2.resize(image, self.config['general']['image_size'])
        
        return feature.hog(
            image,
            orientations=self.config['hog']['orientations'],
            pixels_per_cell=self.config['hog']['pixels_per_cell'],
            cells_per_block=self.config['hog']['cells_per_block'],
            block_norm=self.config['hog']['block_norm'],
            transform_sqrt=self.config['hog']['transform_sqrt'],
            feature_vector=self.config['hog']['feature_vector']
        )
    
    def extract_lbp(self, image):
        """Extrai características LBP"""
        if image.shape != self.config['general']['image_size']:
            image = cv2.resize(image, self.config['general']['image_size'])
        
        lbp = feature.local_binary_pattern(
            image,
            self.config['lbp']['num_points'],
            self.config['lbp']['radius'],
            method=self.config['lbp']['method']
        )
        
        n_bins = self.config['lbp']['num_points'] + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        return hist
    
    def extract_all(self, image):
        """Extrai todas as características"""
        hog = self.extract_hog(image)
        lbp = self.extract_lbp(image)
        combined = np.concatenate([hog, lbp])
        return {'hog': hog, 'lbp': lbp, 'combined': combined}
    
    def process(self):
        """Processa todas as imagens com labels corretas"""
        print("="*60)
        print("EXTRACAO DE CARACTERISTICAS COM LABELS CORRETAS")
        print("="*60)
        
        # Configuracoes
        image_dir = self.config['general']['image_dir']
        identity_file = self.config['general']['identity_file']
        min_images = self.config['general']['min_images_per_person']
        max_persons = self.config['general']['max_persons']
        output_dir = '../data/features'
        
        # Criar diretorios de saida
        for subdir in ['hog', 'lbp', 'combined', 'metadata', 'verification_pairs']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        # 1. Carregar mapeamento
        mapping, counts = self.load_identity_mapping(identity_file, image_dir)
        
        # 2. Filtrar pessoas
        filtered_mapping, filtered_counts = self.filter_persons(
            mapping, counts, min_images, max_persons
        )
        
        if not filtered_mapping:
            raise ValueError("Nenhuma pessoa valida encontrada apos filtro!")
        
        # 3. Mapear person_id para label numerico consecutivo
        unique_persons = sorted(set(filtered_mapping.values()))
        person_to_label = {person_id: idx for idx, person_id in enumerate(unique_persons)}
        
        # 4. Inicializar estruturas para armazenamento
        features_dict = {
            'hog': [],
            'lbp': [],
            'combined': [],
            'image_paths': [],
            'person_ids': [],  # IDs originais do CelebA
            'labels': [],      # Labels numericos consecutivos
            'filenames': []
        }
        
        # 5. Processar imagens
        print(f"\nProcessando {len(filtered_mapping)} imagens...")
        processed = 0
        skipped = 0
        
        for filename, person_id in tqdm(filtered_mapping.items(), desc="Extraindo"):
            try:
                img_path = os.path.join(image_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    skipped += 1
                    continue
                
                # Extrair características
                features = self.extract_all(img)
                
                # Armazenar
                features_dict['hog'].append(features['hog'])
                features_dict['lbp'].append(features['lbp'])
                features_dict['combined'].append(features['combined'])
                features_dict['image_paths'].append(img_path)
                features_dict['person_ids'].append(person_id)
                features_dict['labels'].append(person_to_label[person_id])
                features_dict['filenames'].append(filename)
                
                processed += 1
                
            except Exception as e:
                skipped += 1
                continue
        
        print(f"\nPROCESSAMENTO CONCLUIDO:")
        print(f"  Imagens processadas: {processed}")
        print(f"  Imagens ignoradas: {skipped}")
        
        # 6. Converter para arrays numpy
        for key in ['hog', 'lbp', 'combined']:
            features_dict[key] = np.array(features_dict[key])
        
        features_dict['labels'] = np.array(features_dict['labels'])
        features_dict['person_ids'] = np.array(features_dict['person_ids'])
        
        # 7. Normalizar características
        if self.config['general']['normalize_features']:
            print("\nNormalizando características...")
            features_dict['hog'] = self.scaler_hog.fit_transform(features_dict['hog'])
            features_dict['lbp'] = self.scaler_lbp.fit_transform(features_dict['lbp'])
            features_dict['combined'] = self.scaler_combined.fit_transform(features_dict['combined'])
        
        # 8. Salvar dados
        print("\nSalvando dados...")
        
        # Características
        np.save(os.path.join(output_dir, 'hog', 'features.npy'), features_dict['hog'])
        np.save(os.path.join(output_dir, 'lbp', 'features.npy'), features_dict['lbp'])
        np.save(os.path.join(output_dir, 'combined', 'features.npy'), features_dict['combined'])
        
        # Labels e metadados
        np.save(os.path.join(output_dir, 'metadata', 'labels.npy'), features_dict['labels'])
        np.save(os.path.join(output_dir, 'metadata', 'person_ids.npy'), features_dict['person_ids'])
        np.save(os.path.join(output_dir, 'metadata', 'image_paths.npy'), features_dict['image_paths'])
        np.save(os.path.join(output_dir, 'metadata', 'filenames.npy'), features_dict['filenames'])
        
        # Mapeamentos
        with open(os.path.join(output_dir, 'metadata', 'person_to_label.json'), 'w') as f:
            json.dump(person_to_label, f, indent=2)
        
        with open(os.path.join(output_dir, 'metadata', 'label_to_person.json'), 'w') as f:
            label_to_person = {v: k for k, v in person_to_label.items()}
            json.dump(label_to_person, f, indent=2)
        
        # Escaladores
        if self.config['general']['save_scalers']:
            joblib.dump(self.scaler_hog, os.path.join(output_dir, 'hog', 'scaler.pkl'))
            joblib.dump(self.scaler_lbp, os.path.join(output_dir, 'lbp', 'scaler.pkl'))
            joblib.dump(self.scaler_combined, os.path.join(output_dir, 'combined', 'scaler.pkl'))
        
        # Configurações
        config_output = os.path.join(output_dir, 'extraction_config.yaml')
        with open(config_output, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # 9. Criar pares para verificação facial
        self.create_verification_pairs(features_dict, output_dir)
        
        # 10. Estatísticas finais
        stats = self._calculate_statistics(features_dict, filtered_counts)
        self._save_statistics(stats, output_dir)
        self._print_statistics(stats)
        
        return stats
    
    def create_verification_pairs(self, features_dict, output_dir):
        """Cria pares para tarefa de verificação facial"""
        print("\nCriando pares para verificação facial...")
        
        labels = features_dict['labels']
        n_samples = len(labels)
        
        # Para cada pessoa, criar pares positivos (mesma pessoa)
        positive_pairs = []
        negative_pairs = []
        
        # Agrupar indices por pessoa
        from collections import defaultdict
        indices_by_label = defaultdict(list)
        for idx, label in enumerate(labels):
            indices_by_label[label].append(idx)
        
        # Criar pares positivos (máximo 5 por pessoa)
        for label, indices in indices_by_label.items():
            if len(indices) >= 2:
                # Pegar combinações de 2 em 2
                import itertools
                combinations = list(itertools.combinations(indices, 2))
                # Limitar a 5 pares por pessoa para balancear
                for i, j in combinations[:5]:
                    positive_pairs.append((i, j))
        
        # Criar pares negativos (pessoas diferentes)
        # Pegar um número similar de pares negativos
        n_negatives = len(positive_pairs)
        all_labels = list(indices_by_label.keys())
        
        for _ in range(n_negatives):
            # Escolher duas pessoas diferentes
            label1, label2 = np.random.choice(all_labels, 2, replace=False)
            # Escolher uma imagem de cada pessoa
            idx1 = np.random.choice(indices_by_label[label1])
            idx2 = np.random.choice(indices_by_label[label2])
            negative_pairs.append((idx1, idx2))
        
        # Combinar pares
        all_pairs = positive_pairs + negative_pairs
        pair_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        
        # Embaralhar
        indices = np.random.permutation(len(all_pairs))
        all_pairs = [all_pairs[i] for i in indices]
        pair_labels = [pair_labels[i] for i in indices]
        
        # Salvar
        np.save(os.path.join(output_dir, 'verification_pairs', 'pairs.npy'), all_pairs)
        np.save(os.path.join(output_dir, 'verification_pairs', 'pair_labels.npy'), pair_labels)
        
        print(f"  Pares positivos: {len(positive_pairs)}")
        print(f"  Pares negativos: {len(negative_pairs)}")
        print(f"  Total de pares: {len(all_pairs)}")
    
    def _calculate_statistics(self, features_dict, counts):
        """Calcula estatísticas finais"""
        unique_labels = np.unique(features_dict['labels'])
        label_counts = Counter(features_dict['labels'])
        
        stats = {
            'total_images': len(features_dict['labels']),
            'unique_persons': len(unique_labels),
            'hog_feature_dimension': features_dict['hog'].shape[1],
            'lbp_feature_dimension': features_dict['lbp'].shape[1],
            'combined_feature_dimension': features_dict['combined'].shape[1],
            'hog_features_mean': float(np.mean(features_dict['hog'])),
            'hog_features_std': float(np.std(features_dict['hog'])),
            'lbp_features_mean': float(np.mean(features_dict['lbp'])),
            'lbp_features_std': float(np.std(features_dict['lbp'])),
            'images_per_person_min': int(min(label_counts.values())),
            'images_per_person_max': int(max(label_counts.values())),
            'images_per_person_avg': float(np.mean(list(label_counts.values()))),
            'images_per_person_std': float(np.std(list(label_counts.values())))
        }
        
        return stats
    
    def _save_statistics(self, stats, output_dir):
        """Salva estatísticas em arquivo"""
        stats_path = os.path.join(output_dir, 'metadata', 'statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("ESTATISTICAS DA EXTRACAO COM LABELS CORRETAS\n")
            f.write("="*60 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    def _print_statistics(self, stats):
        """Imprime estatísticas"""
        print("\n" + "="*60)
        print("ESTATISTICAS FINAIS")
        print("="*60)
        
        print(f"\nDADOS GERAIS:")
        print(f"  Total de imagens: {stats['total_images']}")
        print(f"  Pessoas únicas: {stats['unique_persons']}")
        print(f"  Imagens por pessoa (média): {stats['images_per_person_avg']:.1f}")
        print(f"  Imagens por pessoa (mín): {stats['images_per_person_min']}")
        print(f"  Imagens por pessoa (máx): {stats['images_per_person_max']}")
        
        print(f"\nCARACTERÍSTICAS:")
        print(f"  HOG: {stats['hog_feature_dimension']} dimensões")
        print(f"  LBP: {stats['lbp_feature_dimension']} dimensões")
        print(f"  Combinado: {stats['combined_feature_dimension']} dimensões")

def main():
    parser = argparse.ArgumentParser(description='Extração com labels corretas do CelebA')
    parser.add_argument('--config', default='config/extraction_config.yaml',
                       help='Arquivo de configuração')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, args.config)
    
    # Inicializar e executar
    extractor = FeatureExtractorWithLabels(config_path)
    stats = extractor.process()
    
    print("\n" + "="*60)
    print("EXTRACAO CONCLUIDA COM SUCESSO!")
    print("="*60)
    print("Próximos passos:")
    print("1. Dados para IDENTIFICAÇÃO: ../data/features/")
    print("2. Dados para AUTENTICAÇÃO: ../data/features/verification_pairs/")
    print("3. Agora pode treinar MLP e SVM")
    print("="*60)

if __name__ == "__main__":
    main()