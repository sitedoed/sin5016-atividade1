#!/usr/bin/env python3
"""
copy_selected_images.py
Copia as imagens selecionadas (com base nos IDs) para Images/Selected_images/

Uso:
    python copy_selected_images.py
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import sys

def read_selected_ids(selected_ids_file):
    """Lê os IDs selecionados do arquivo"""
    selected_ids = set()
    
    print(f"Lendo IDs selecionados: {selected_ids_file}")
    
    with open(selected_ids_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if parts:
                    selected_ids.add(parts[0])  # Apenas o ID
    
    print(f"  IDs lidos: {len(selected_ids)}")
    return selected_ids

def read_image_mappings(identity_file, selected_ids):
    """Lê mapeamentos de imagens e filtra pelos IDs selecionados"""
    print(f"Lendo mapeamentos de imagens: {identity_file}")
    
    images_to_copy = []
    images_by_id = {}
    
    with open(identity_file, 'r') as f:
        lines_processed = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, person_id = parts
                
                if person_id in selected_ids:
                    images_to_copy.append(image_name)
                    
                    # Agrupar por ID (útil para estatísticas)
                    if person_id not in images_by_id:
                        images_by_id[person_id] = []
                    images_by_id[person_id].append(image_name)
                
                lines_processed += 1
                if lines_processed % 50000 == 0:
                    print(f"  Processadas {lines_processed:,} linhas...")
    
    print(f"  Imagens encontradas para IDs selecionados: {len(images_to_copy):,}")
    print(f"  IDs com imagens: {len(images_by_id)}")
    
    return images_to_copy, images_by_id

def copy_images_to_directory(images_to_copy, source_dir, dest_dir, test_mode=False):
    """Copia as imagens para o diretório destino"""
    # Criar diretório destino
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if test_mode:
        images_to_copy = images_to_copy[:50]  # Apenas 50 para teste
        print(f"Modo teste: copiando apenas {len(images_to_copy)} imagens")
    
    print(f"\nCopiando imagens:")
    print(f"  De: {source_dir}")
    print(f"  Para: {dest_dir}")
    print(f"  Total: {len(images_to_copy):,} imagens")
    
    copied_count = 0
    failed_count = 0
    missing_count = 0
    
    start_time = datetime.now()
    
    for i, image_name in enumerate(images_to_copy, 1):
        source_path = Path(source_dir) / image_name
        dest_path_file = dest_path / image_name
        
        # Verificar se imagem existe
        if not source_path.exists():
            missing_count += 1
            if missing_count <= 5:
                print(f"  [AVISO] Imagem não encontrada: {image_name}")
            continue
        
        try:
            # Copiar imagem
            shutil.copy2(source_path, dest_path_file)
            copied_count += 1
            
            # Progresso
            if i % 1000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = i / elapsed if elapsed > 0 else 0
                print(f"  Progresso: {i:,}/{len(images_to_copy):,} "
                      f"({i/len(images_to_copy)*100:.1f}%) - "
                      f"{speed:.1f} img/seg")
                
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"  [ERRO] Erro ao copiar {image_name}: {e}")
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    return copied_count, failed_count, missing_count, execution_time

def create_sample_structure(images_by_id, dest_dir, samples_per_id=2):
    """Cria estrutura de amostra organizada por ID"""
    sample_dir = Path(dest_dir) / "Sample_by_ID"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCriando amostra organizada por ID em: {sample_dir}")
    
    for person_id, images in images_by_id.items():
        id_dir = sample_dir / f"ID_{person_id}"
        id_dir.mkdir(exist_ok=True)
        
        # Copiar algumas imagens de amostra
        for img_name in images[:samples_per_id]:
            source = Path(dest_dir) / img_name
            if source.exists():
                shutil.copy2(source, id_dir / img_name)
    
    print(f"  Amostra criada com {len(images_by_id)} diretórios de ID")

def verify_copy(dest_dir, expected_count):
    """Verifica se todas as imagens foram copiadas"""
    print(f"\nVerificando cópia em: {dest_dir}")
    
    dest_path = Path(dest_dir)
    
    # Contar arquivos .jpg no diretório
    image_files = list(dest_path.glob('*.jpg'))
    actual_count = len(image_files)
    
    print(f"Imagens encontradas: {actual_count:,}")
    print(f"Imagens esperadas: {expected_count:,}")
    
    if actual_count == expected_count:
        print("✅ VERIFICAÇÃO: CORRETO! Todas imagens copiadas.")
        return True
    elif abs(actual_count - expected_count) <= 10:
        print(f"⚠️  VERIFICAÇÃO: QUASE! Diferença de {abs(actual_count - expected_count)} imagens.")
        return True
    else:
        print(f"❌ VERIFICAÇÃO: PROBLEMA! Diferença de {abs(actual_count - expected_count)} imagens.")
        return False

def save_statistics(selected_ids_file, source_dir, dest_dir, 
                    images_to_copy, images_by_id,
                    copied_count, failed_count, missing_count, 
                    execution_time):
    """Salva estatísticas da cópia"""
    stats_file = Path(dest_dir) / "copy_statistics.txt"
    
    total_attempted = copied_count + failed_count + missing_count
    
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ESTATÍSTICAS DA CÓPIA DE IMAGENS SELECIONADAS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Data/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arquivo de IDs: {selected_ids_file}\n")
        f.write(f"Arquivo de mapeamento: data/raw/identity_CelebA.txt\n")
        f.write(f"Diretório fonte: {source_dir}\n")
        f.write(f"Diretório destino: {dest_dir}\n\n")
        
        f.write("RESULTADOS:\n")
        f.write(f"  - IDs selecionados: {len(images_by_id):,}\n")
        f.write(f"  - Imagens para copiar: {len(images_to_copy):,}\n")
        f.write(f"  - Imagens copiadas com sucesso: {copied_count:,}\n")
        f.write(f"  - Imagens não encontradas: {missing_count:,}\n")
        f.write(f"  - Imagens com erro na cópia: {failed_count:,}\n")
        f.write(f"  - Taxa de sucesso: {(copied_count/len(images_to_copy))*100:.2f}%\n")
        f.write(f"  - Tempo total: {execution_time:.2f} segundos\n")
        f.write(f"  - Velocidade: {len(images_to_copy)/execution_time:.1f} imagens/segundo\n\n")
        
        f.write("DISTRIBUIÇÃO POR ID:\n")
        # Ordenar IDs por número de imagens
        sorted_ids = sorted(images_by_id.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)
        
        for i, (person_id, images) in enumerate(sorted_ids[:20], 1):
            f.write(f"  {i:3}. ID {person_id:6}: {len(images):3} imagens\n")
        
        if len(sorted_ids) > 20:
            f.write(f"  ... e mais {len(sorted_ids)-20} IDs\n\n")
        
        f.write("USO DE DISCO (estimado):\n")
        avg_size_kb = 45  # Tamanho médio aproximado das imagens CelebA
        total_size_mb = (copied_count * avg_size_kb) / 1024
        total_size_gb = total_size_mb / 1024
        
        f.write(f"  - Tamanho médio por imagem: ~{avg_size_kb} KB\n")
        f.write(f"  - Espaço total estimado: {total_size_mb:,.1f} MB ({total_size_gb:.2f} GB)\n\n")
        
        f.write("PRIMEIRAS 20 IMAGENS COPIADAS:\n")
        for i, img_name in enumerate(images_to_copy[:20], 1):
            f.write(f"  {i:3}. {img_name}\n")
    
    print(f"Estatísticas salvas em: {stats_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Copia imagens selecionadas (por ID) para Images/Selected_images/'
    )
    
    parser.add_argument('--selected_ids', type=str,
                       default='data/processed/selected_ids_limited.txt',
                       help='Arquivo com IDs selecionados')
    
    parser.add_argument('--identity_file', type=str,
                       default='data/raw/identity_CelebA.txt',
                       help='Arquivo identity_CelebA.txt com mapeamentos')
    
    parser.add_argument('--source_dir', type=str,
                       default='Images/Original_images/img_align_celeba',  # ← CAMINHO CORRETO AQUI
                       help='Diretório fonte das imagens')
    
    parser.add_argument('--dest_dir', type=str,
                       default='Images/Selected_images',
                       help='Diretório destino')
    
    parser.add_argument('--test_mode', action='store_true',
                       help='Modo teste (copia apenas 50 imagens)')
    
    parser.add_argument('--create_sample', action='store_true',
                       help='Criar amostra organizada por ID')
    
    parser.add_argument('--skip_verify', action='store_true',
                       help='Pular verificação após cópia')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CÓPIA DE IMAGENS POR IDS SELECIONADOS")
    print("="*60)
    
    # Verificar arquivos necessários
    required_files = [
        args.selected_ids,
        args.identity_file
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"[ERRO] Arquivo não encontrado: {file_path}")
            sys.exit(1)
    
    # Verificar diretório fonte
    if not Path(args.source_dir).exists():
        print(f"[ERRO] Diretório fonte não encontrado: {args.source_dir}")
        print("Certifique-se de que as imagens do CelebA estão disponíveis.")
        sys.exit(1)
    
    # 1. Ler IDs selecionados
    selected_ids = read_selected_ids(args.selected_ids)
    
    # 2. Ler mapeamentos e filtrar imagens
    images_to_copy, images_by_id = read_image_mappings(
        args.identity_file, selected_ids
    )
    
    if len(images_to_copy) == 0:
        print("[ERRO] Nenhuma imagem encontrada para os IDs selecionados!")
        sys.exit(1)
    
    # 3. Copiar imagens
    copied, failed, missing, time_taken = copy_images_to_directory(
        images_to_copy, args.source_dir, args.dest_dir, args.test_mode
    )
    
    # 4. Mostrar resultados
    print(f"\n" + "="*60)
    print("RESUMO DA CÓPIA")
    print("="*60)
    print(f"IDs selecionados: {len(images_by_id):,}")
    print(f"Imagens para copiar: {len(images_to_copy):,}")
    print(f"Imagens copiadas: {copied:,} ({copied/len(images_to_copy)*100:.1f}%)")
    print(f"Imagens não encontradas: {missing:,} ({missing/len(images_to_copy)*100:.1f}%)")
    print(f"Imagens com erro: {failed:,} ({failed/len(images_to_copy)*100:.1f}%)")
    print(f"Tempo total: {time_taken:.2f} segundos")
    print(f"Velocidade: {len(images_to_copy)/time_taken:.1f} imagens/segundo")
    
    # Estimar espaço em disco
    if copied > 0:
        estimated_gb = (copied * 45) / 1024 / 1024
        print(f"Espaço estimado: {estimated_gb:.2f} GB")
    
    # 5. Salvar estatísticas
    save_statistics(
        args.selected_ids, args.source_dir, args.dest_dir,
        images_to_copy, images_by_id,
        copied, failed, missing, time_taken
    )
    
    # 6. Criar amostra organizada (opcional)
    if args.create_sample and not args.test_mode:
        create_sample_structure(images_by_id, args.dest_dir)
    
    # 7. Verificar cópia
    if not args.skip_verify and not args.test_mode:
        expected = copied  # Esperamos ter copiado todas as que existiam
        verify_copy(args.dest_dir, expected)
    
    print("\n" + "="*60)
    print("CÓPIA CONCLUÍDA!")
    print("="*60)
    
    # Sugestões para próximos passos
    print(f"\nPRÓXIMOS PASSOS:")
    print(f"  1. Verificar amostra: ls {args.dest_dir} | head -20")
    print(f"  2. Contar imagens: ls {args.dest_dir}/*.jpg | wc -l")
    print(f"  3. Extrair features HOG: python Codigo/extract_hog_features.py")

if __name__ == "__main__":
    main()