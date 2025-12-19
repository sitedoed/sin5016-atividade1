# Codigo/check_structure.py
import os
import numpy as np

def check_image_structure():
    """Verifica como as imagens estão organizadas"""
    base_path = "../Images/Selected_images"
    
    print("Analisando estrutura de diretórios...")
    print("="*60)
    
    # Verificar se é uma estrutura plana ou hierárquica
    items = os.listdir(base_path)
    
    # Contar quantos são diretórios e quantos são arquivos
    dirs = [d for d in items if os.path.isdir(os.path.join(base_path, d))]
    files = [f for f in items if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Total de itens em {base_path}: {len(items)}")
    print(f"Diretórios: {len(dirs)}")
    print(f"Arquivos de imagem: {len(files)}")
    
    if dirs:
        print("\n📁 Estrutura com diretórios por pessoa:")
        # Contar imagens nos primeiros 5 diretórios
        for d in dirs[:5]:
            dir_path = os.path.join(base_path, d)
            imgs = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {d}: {len(imgs)} imagens")
            
        # Verificar total
        total_imgs = 0
        for d in dirs:
            dir_path = os.path.join(base_path, d)
            imgs = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_imgs += len(imgs)
        print(f"\nTotal estimado de imagens: {total_imgs}")
        
    elif files:
        print("\n📄 Estrutura plana - todas as imagens em um diretório:")
        print(f"Primeiros 5 arquivos: {files[:5]}")
        
        # Tentar extrair padrões dos nomes
        print("\n📊 Padrões nos nomes dos arquivos:")
        for f in files[:10]:
            print(f"  {f}")
    
    return dirs, files

if __name__ == "__main__":
    dirs, files = check_image_structure()