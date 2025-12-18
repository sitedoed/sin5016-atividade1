#!/usr/bin/env python3
"""
filter_images.py - VERSÃO CORRIGIDA
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

def read_selected_ids(selected_ids_file):
    """Lê os IDs do arquivo de IDs selecionados"""
    selected_ids = set()
    
    print(f"Lendo IDs selecionados de: {selected_ids_file}")
    
    try:
        with open(selected_ids_file, 'r') as f:
            lines_read = 0
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if parts:
                        selected_ids.add(int(parts[0]))
                        lines_read += 1
                        
                        if lines_read % 500 == 0:
                            print(f"  Lidos {lines_read} IDs...")
        
        print(f"  Total: {len(selected_ids)} IDs lidos")
        
    except Exception as e:
        print(f"[ERRO] Erro ao ler arquivo de IDs: {e}")
        sys.exit(1)
    
    return selected_ids, len(selected_ids)

def filter_images(original_mapping_file, selected_ids, output_file):
    """Filtra o arquivo de mapeamento original"""
    print(f"\nFiltrando imagens de: {original_mapping_file}")
    print(f"Salvando em: {output_file}")
    
    start_time = datetime.now()
    filtered_count = 0
    total_processed = 0
    
    try:
        with open(original_mapping_file, 'r') as fin, \
             open(output_file, 'w') as fout:
            
            # Escrever cabeçalho
            fout.write(f"# Arquivo filtrado gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fout.write(f"# IDs selecionados: {len(selected_ids)}\n")
            fout.write(f"# Arquivo original: {original_mapping_file}\n")
            fout.write(f"# Formato: nome_imagem ID_pessoa\n")
            fout.write("#" * 80 + "\n\n")
            
            for line in fin:
                total_processed += 1
                
                if total_processed % 50000 == 0:
                    print(f"  Processadas {total_processed:,} linhas, filtradas {filtered_count:,} imagens...")
                
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, person_id = parts
                    
                    if int(person_id) in selected_ids:
                        fout.write(line)
                        filtered_count += 1
    
    except Exception as e:
        print(f"[ERRO] Erro ao filtrar imagens: {e}")
        sys.exit(1)
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    return filtered_count, execution_time, total_processed

def save_statistics(selected_ids_file, output_file, filtered_count, 
                   execution_time, total_processed, selected_ids_count):
    """Salva estatísticas do processo de filtragem"""
    stats_file = output_file.replace('.txt', '_stats.txt')
    
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ESTATISTICAS DO FILTRO DE IMAGENS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Data/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arquivo de IDs: {selected_ids_file}\n")
        f.write(f"Arquivo original: data/raw/identity_CelebA.txt\n")
        f.write(f"Arquivo de saida: {output_file}\n\n")
        
        f.write("RESULTADOS:\n")
        f.write(f"  - IDs selecionados: {selected_ids_count}\n")
        f.write(f"  - Linhas processadas: {total_processed:,}\n")
        f.write(f"  - Imagens filtradas: {filtered_count:,}\n")
        f.write(f"  - Imagens descartadas: {total_processed - filtered_count:,}\n")
        f.write(f"  - Taxa de retencao: {(filtered_count/total_processed)*100:.2f}%\n")
        f.write(f"  - Tempo de execucao: {execution_time:.2f} segundos\n")
        f.write(f"  - Velocidade: {total_processed/execution_time:.0f} linhas/segundo\n\n")
        
        f.write("PRIMEIRAS 20 IMAGENS FILTRADAS:\n")
        try:
            with open(output_file, 'r') as img_file:
                lines = []
                for i, line in enumerate(img_file):
                    if i >= 10:  # Pular cabeçalho
                        lines.append(line.strip())
                    if len(lines) >= 20:
                        break
                
                for i, line in enumerate(lines, 1):
                    if line and not line.startswith('#'):
                        f.write(f"  {i:3}. {line}\n")
        except:
            f.write("  [Nao foi possivel ler o arquivo de saida]\n")
    
    print(f"Estatisticas salvas em: {stats_file}")

def verify_filtering(output_file, expected_count=50648):
    """Verifica se a filtragem foi realizada corretamente"""
    print("\n" + "="*60)
    print("VERIFICACAO DO RESULTADO")
    print("="*60)
    
    try:
        actual_count = 0
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        actual_count += 1
        
        print(f"Imagens encontradas no arquivo filtrado: {actual_count:,}")
        print(f"Imagens esperadas: {expected_count:,}")
        
        if actual_count == expected_count:
            print("✅ VERIFICACAO: CORRETO! Contagem bate com o esperado.")
        elif abs(actual_count - expected_count) <= 10:
            print(f"⚠️  VERIFICACAO: QUASE! Diferenca de {abs(actual_count - expected_count)} imagens.")
        else:
            print(f"❌ VERIFICACAO: PROBLEMA! Diferenca de {abs(actual_count - expected_count)} imagens.")
    
    except Exception as e:
        print(f"[ERRO] Na verificacao: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Filtra imagens do CelebA baseado em IDs selecionados'
    )
    
    parser.add_argument('--selected_ids', type=str,
                       default='data/processed/selected_ids_limited.txt',
                       help='Arquivo com IDs selecionados')
    
    parser.add_argument('--original_mapping', type=str,
                       default='data/raw/identity_CelebA.txt',
                       help='Arquivo original de mapeamento')
    
    parser.add_argument('--output', type=str,
                       default='data/processed/filtered_images.txt',
                       help='Arquivo de saida com imagens filtradas')
    
    parser.add_argument('--no_stats', action='store_true',
                       help='Nao gerar arquivo de estatisticas')
    
    parser.add_argument('--no_verify', action='store_true',
                       help='Nao verificar resultado final')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FILTRAGEM DE IMAGENS - CelebA Dataset")
    print("="*60)
    
    # 1. Garantir que diretórios existem
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Ler IDs selecionados
    selected_ids, selected_ids_count = read_selected_ids(args.selected_ids)
    
    # 3. Filtrar imagens
    filtered_count, exec_time, total_processed = filter_images(
        args.original_mapping, selected_ids, args.output
    )
    
    # 4. Mostrar resultados
    print(f"\n" + "="*60)
    print("RESULTADO DA FILTRAGEM")
    print("="*60)
    print(f"IDs selecionados: {selected_ids_count:,}")
    print(f"Imagens originais processadas: {total_processed:,}")
    print(f"Imagens filtradas: {filtered_count:,}")
    print(f"Imagens descartadas: {total_processed - filtered_count:,}")
    print(f"Taxa de retencao: {(filtered_count/total_processed)*100:.2f}%")
    print(f"Tempo total: {exec_time:.2f} segundos")
    print(f"Arquivo gerado: {args.output}")
    
    # 5. Salvar estatísticas
    if not args.no_stats:
        save_statistics(args.selected_ids, args.output, filtered_count, 
                       exec_time, total_processed, selected_ids_count)
    
    # 6. Verificar
    if not args.no_verify:
        verify_filtering(args.output, 50648)
    
    print("\n" + "="*60)
    print("FILTRAGEM CONCLUIDA!")
    print("="*60)
    

if __name__ == "__main__":
    main()