#!/usr/bin/env python3
"""
select_ids_with_limit.py
Seleciona IDs ordenados por frequência até atingir limite percentual das imagens.

Uso:
    python select_ids_with_limit.py --min_images 30 --max_percentage 25
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

def load_identity_file(identity_file):
    """Carrega o arquivo identity_CelebA.txt e conta frequências"""
    id_counts = Counter()
    
    print(f"Carregando {identity_file}...")
    
    try:
        with open(identity_file, 'r') as f:
            total_lines = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, person_id = parts
                    id_counts[int(person_id)] += 1
                    total_lines += 1
                    
                    if total_lines % 50000 == 0:
                        print(f"  Processadas {total_lines:,} linhas...")
                
        print(f"  Concluído! {total_lines:,} imagens processadas.")
        
    except Exception as e:
        print(f"[ERRO] Erro ao ler arquivo: {e}")
        sys.exit(1)
    
    return id_counts, total_lines

def select_ids_with_limits(id_counts, total_images, min_images=30, max_percentage=25):
    """
    Seleciona IDs ordenados por frequência até atingir limite percentual
    
    Args:
        id_counts (Counter): Contagem de imagens por ID
        total_images (int): Total de imagens no dataset
        min_images (int): Mínimo de imagens por ID
        max_percentage (float): Percentual máximo de imagens a selecionar
        
    Returns:
        tuple: (selected_ids, accumulated_images, percentage)
    """
    # 1. Filtrar IDs com pelo menos min_images
    filtered_ids = [(pid, count) for pid, count in id_counts.items() if count >= min_images]
    
    # 2. Ordenar por número de imagens (decrescente)
    filtered_ids.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Calcular limite absoluto
    max_images = total_images * max_percentage / 100
    print(f"\nLimites:")
    print(f"  - Total imagens dataset: {total_images:,}")
    print(f"  - Percentual máximo: {max_percentage}%")
    print(f"  - Limite absoluto: {max_images:,.0f} imagens")
    print(f"  - Mínimo por ID: {min_images} imagens")
    
    # 4. Selecionar até atingir o limite
    selected_ids = []
    accumulated_images = 0
    accumulated_percentage = 0
    
    print(f"\nSelecionando IDs...")
    for i, (pid, count) in enumerate(filtered_ids, 1):
        # Verificar se adicionar este ID ultrapassaria o limite
        if accumulated_images + count > max_images:
            print(f"  Limite atingido! Parando na seleção {i-1} IDs")
            break
            
        selected_ids.append((pid, count))
        accumulated_images += count
        accumulated_percentage = (accumulated_images / total_images) * 100
        
        # Progresso a cada 100 IDs
        if i % 100 == 0:
            print(f"  Selecionados {i} IDs, {accumulated_images:,} imagens ({accumulated_percentage:.1f}%)")
    
    return selected_ids, accumulated_images, accumulated_percentage

def print_selection_summary(selected_ids, total_ids, total_images, accumulated_images, accumulated_percentage):
    """Imprime resumo da seleção"""
    print("\n" + "="*60)
    print("RESUMO DA SELECAO")
    print("="*60)
    
    print(f"\nCRITERIOS ATENDIDOS:")
    print(f"  - IDs com >= {min_images} imagens")
    print(f"  - Parou quando imagens > {max_percentage}% do dataset")
    
    print(f"\nRESULTADO:")
    print(f"  - IDs selecionados: {len(selected_ids):,} / {total_ids:,}")
    print(f"  - Imagens selecionadas: {accumulated_images:,} / {total_images:,}")
    print(f"  - Percentual de imagens: {accumulated_percentage:.2f}%")
    
    if selected_ids:
        avg_images = accumulated_images / len(selected_ids)
        print(f"  - Media imagens/ID: {avg_images:.2f}")
        print(f"  - ID com mais imagens: {selected_ids[0][0]} ({selected_ids[0][1]} imagens)")
        print(f"  - ID com menos imagens: {selected_ids[-1][0]} ({selected_ids[-1][1]} imagens)")
        
        print(f"\nTOP 10 IDs SELECIONADOS:")
        for i, (pid, count) in enumerate(selected_ids[:10], 1):
            print(f"    {i:2}. ID {pid:6}: {count:3} imagens")
        
        # Distribuição
        print(f"\nDISTRIBUICAO DOS IDs SELECIONADOS:")
        bins = [30, 31, 32, 33, 34, 35]
        for i in range(len(bins)-1):
            min_b, max_b = bins[i], bins[i+1]
            count = sum(1 for _, img_count in selected_ids if min_b <= img_count < max_b)
            if count > 0:
                print(f"  - {min_b}-{max_b-1} imagens: {count:4} IDs")
        
        count_35 = sum(1 for _, img_count in selected_ids if img_count == 35)
        if count_35 > 0:
            print(f"  - 35 imagens: {count_35:4} IDs")
    
    print("\n" + "="*60)

def save_results(selected_ids, output_file, total_images, accumulated_percentage):
    """Salva os resultados em arquivo"""
    # Garantir diretório
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salvar IDs
    with open(output_file, 'w') as f:
        f.write(f"# IDs selecionados com limite de {max_percentage}% das imagens\n")
        f.write(f"# Total imagens dataset: {total_images}\n")
        f.write(f"# Imagens selecionadas: {sum(count for _, count in selected_ids)}\n")
        f.write(f"# Percentual: {accumulated_percentage:.2f}%\n")
        f.write(f"# Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for person_id, count in selected_ids:
            f.write(f"{person_id}\t{count}\n")
    
    print(f"Arquivo salvo: {output_file}")
    
    # Salvar metadados
    metadata_file = output_file.replace('.txt', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("METADADOS DA SELECAO COM LIMITE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Arquivo de saida: {output_file}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CRITERIOS:\n")
        f.write(f"  - Minimo imagens por ID: {min_images}\n")
        f.write(f"  - Maximo percentual de imagens: {max_percentage}%\n")
        f.write(f"  - Limite absoluto de imagens: {total_images * max_percentage / 100:.0f}\n\n")
        
        f.write("RESULTADOS:\n")
        f.write(f"  - IDs selecionados: {len(selected_ids)}\n")
        f.write(f"  - Imagens selecionadas: {sum(count for _, count in selected_ids)}\n")
        f.write(f"  - Percentual do dataset: {accumulated_percentage:.2f}%\n")
        f.write(f"  - Media imagens/ID: {sum(count for _, count in selected_ids)/len(selected_ids):.2f}\n\n")
        
        f.write("DISTRIBUICAO:\n")
        for i, (pid, count) in enumerate(selected_ids[:50], 1):
            f.write(f"{i:3}. ID {pid:6}: {count:3} imagens\n")
        
        if len(selected_ids) > 50:
            f.write(f"... e mais {len(selected_ids)-50} IDs\n")
    
    print(f"Metadados salvos: {metadata_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Seleciona IDs ordenados por frequência até limite percentual',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--identity_file', type=str, 
                       default='data/raw/identity_CelebA.txt',
                       help='Arquivo identity_CelebA.txt')
    
    parser.add_argument('--min_images', type=int, default=30,
                       help='Minimo de imagens por ID (padrao: 30)')
    
    parser.add_argument('--max_percentage', type=float, default=25,
                       help='Percentual maximo de imagens a selecionar (padrao: 25)')
    
    parser.add_argument('--output_file', type=str,
                       default='data/processed/selected_ids_limited.txt',
                       help='Arquivo de saida')
    
    args = parser.parse_args()
    
    global min_images, max_percentage
    min_images = args.min_images
    max_percentage = args.max_percentage
    
    print("="*60)
    print("SELECAO DE IDs COM LIMITE PERCENTUAL")
    print("="*60)
    
    # Carregar dados
    id_counts, total_images = load_identity_file(args.identity_file)
    total_ids = len(id_counts)
    
    print(f"\nDATASET COMPLETO:")
    print(f"  - IDs unicos: {total_ids:,}")
    print(f"  - Total imagens: {total_images:,}")
    print(f"  - Media geral: {total_images/total_ids:.2f} imagens/ID")
    
    # Selecionar IDs com limites
    selected_ids, accumulated_images, accumulated_percentage = select_ids_with_limits(
        id_counts, total_images, args.min_images, args.max_percentage
    )
    
    # Mostrar resumo
    print_selection_summary(selected_ids, total_ids, total_images, 
                          accumulated_images, accumulated_percentage)
    
    # Salvar resultados
    save_results(selected_ids, args.output_file, total_images, accumulated_percentage)
    
    print("\nSELECAO CONCLUIDA!")
    print("="*60)
    
if __name__ == "__main__":
    from datetime import datetime
    main()