import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def preprocess_and_resize_image(image_path, output_path, target_size=(64, 64)):
    """
    Pré-processa uma imagem: redimensiona para 64x64 e converte para escala de cinza
    
    Args:
        image_path (str): Caminho da imagem original
        output_path (str): Caminho para salvar a imagem processada
        target_size (tuple): Tamanho alvo (largura, altura)
    """
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Erro ao carregar imagem: {image_path}")
            return False
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar para 64x64
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Aplicar equalização de histograma para melhorar contraste
        equalized = cv2.equalizeHist(resized)
        
        # Salvar imagem processada
        cv2.imwrite(output_path, equalized)
        return True
        
    except Exception as e:
        print(f"Erro ao processar {image_path}: {str(e)}")
        return False

def batch_preprocess_images(input_dir, output_dir, target_size=(64, 64)):
    """
    Processa em lote todas as imagens de um diretório
    
    Args:
        input_dir (str): Diretório com imagens originais
        output_dir (str): Diretório para salvar imagens processadas
        target_size (tuple): Tamanho alvo das imagens
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Listar todas as imagens
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Encontradas {len(image_files)} imagens para processar")
    
    # Processar imagens
    success_count = 0
    failed_images = []
    
    for img_path in tqdm(image_files, desc="Processando imagens"):
        # Criar nome do arquivo de saída
        rel_path = os.path.relpath(img_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Criar subdiretórios se necessário
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Processar imagem
        if preprocess_and_resize_image(img_path, output_path, target_size):
            success_count += 1
        else:
            failed_images.append(img_path)
    
    # Gerar relatório
    print("\n" + "="*50)
    print("RELATÓRIO DE PROCESSAMENTO")
    print("="*50)
    print(f"Total de imagens processadas: {success_count}")
    print(f"Total de falhas: {len(failed_images)}")
    
    if failed_images:
        print("\nImagens que falharam no processamento:")
        for img in failed_images[:10]:  # Mostrar apenas as 10 primeiras
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... e mais {len(failed_images) - 10} imagens")
    
    return success_count, failed_images

def verify_processed_images(output_dir, expected_size=(64, 64)):
    """
    Verifica se as imagens processadas estão corretas
    
    Args:
        output_dir (str): Diretório com imagens processadas
        expected_size (tuple): Tamanho esperado das imagens
    """
    print("\n" + "="*50)
    print("VERIFICAÇÃO DAS IMAGENS PROCESSADAS")
    print("="*50)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    processed_files = []
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                processed_files.append(os.path.join(root, file))
    
    print(f"Total de imagens processadas encontradas: {len(processed_files)}")
    
    # Verificar algumas imagens aleatoriamente
    import random
    sample_size = min(10, len(processed_files))
    sample_files = random.sample(processed_files, sample_size)
    
    issues = []
    for img_path in sample_files:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                issues.append(f"Erro ao carregar: {img_path}")
            elif img.shape != expected_size:
                issues.append(f"Tamanho incorreto {img.shape} em: {img_path}")
            # Verificar se é realmente em escala de cinza
            elif len(img.shape) != 2:
                issues.append(f"Não é escala de cinza: {img_path}")
        except Exception as e:
            issues.append(f"Exceção em {img_path}: {str(e)}")
    
    if not issues:
        print("✓ Todas as imagens verificadas estão corretas!")
    else:
        print(f"✗ Foram encontrados {len(issues)} problemas:")
        for issue in issues:
            print(f"  - {issue}")

def main():
    parser = argparse.ArgumentParser(description='Pré-processamento de imagens para reconhecimento facial')
    parser.add_argument('--input_dir', type=str, 
                       default='../Images/Selected_images',
                       help='Diretório com imagens originais')
    parser.add_argument('--output_dir', type=str,
                       default='../Images/Optimized_images',
                       help='Diretório para salvar imagens processadas')
    parser.add_argument('--size', type=int, default=64,
                       help='Tamanho das imagens de saída (quadrado)')
    parser.add_argument('--verify', action='store_true',
                       help='Verificar imagens processadas')
    
    args = parser.parse_args()
    
    # Definir caminhos absolutos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, args.input_dir.lstrip('./'))
    output_dir = os.path.join(base_dir, args.output_dir.lstrip('./'))
    
    if not os.path.exists(input_dir):
        print(f"Erro: Diretório de entrada não encontrado: {input_dir}")
        return
    
    if args.verify:
        # Apenas verificar imagens já processadas
        verify_processed_images(output_dir, (args.size, args.size))
    else:
        # Processar imagens
        print("="*50)
        print("INICIANDO PRÉ-PROCESSAMENTO DE IMAGENS")
        print("="*50)
        print(f"Diretório de entrada: {input_dir}")
        print(f"Diretório de saída: {output_dir}")
        print(f"Tamanho alvo: {args.size}x{args.size} pixels")
        print("="*50)
        
        success_count, failed_images = batch_preprocess_images(
            input_dir, output_dir, (args.size, args.size)
        )
        
        # Salvar lista de falhas se houver
        if failed_images:
            log_dir = os.path.join(base_dir, 'results', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'preprocessing_errors.txt')
            with open(log_file, 'w') as f:
                f.write("Imagens que falharam no processamento:\n")
                for img in failed_images:
                    f.write(f"{img}\n")
            print(f"\nLista de falhas salva em: {log_file}")
        
        # Verificação automática após processamento
        verify_processed_images(output_dir, (args.size, args.size))

if __name__ == "__main__":
    main()