import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_preprocessing(original_dir, processed_dir, num_samples=5):
    """
    Visualiza exemplos antes e depois do pré-processamento
    
    Args:
        original_dir (str): Diretório com imagens originais
        processed_dir (str): Diretório com imagens processadas
        num_samples (int): Número de exemplos para visualizar
    """
    # Encontrar imagens comuns nos dois diretórios
    import glob
    
    original_images = glob.glob(os.path.join(original_dir, "**/*.jpg"), recursive=True)
    if not original_images:
        original_images = glob.glob(os.path.join(original_dir, "**/*.png"), recursive=True)
    
    # Selecionar amostras aleatórias
    import random
    if len(original_images) > num_samples:
        samples = random.sample(original_images, num_samples)
    else:
        samples = original_images
        num_samples = len(original_images)
    
    # Configurar plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, orig_path in enumerate(samples):
        # Obter caminho da imagem processada
        rel_path = os.path.relpath(orig_path, original_dir)
        proc_path = os.path.join(processed_dir, rel_path)
        
        # Carregar imagens
        orig_img = cv2.imread(orig_path)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        proc_img = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)
        
        # Plotar original
        axes[i, 0].imshow(orig_img_rgb)
        axes[i, 0].set_title(f'Original\n{orig_img.shape}')
        axes[i, 0].axis('off')
        
        # Plotar original em cinza
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        axes[i, 1].imshow(orig_gray, cmap='gray')
        axes[i, 1].set_title(f'Original (Gray)\n{orig_gray.shape}')
        axes[i, 1].axis('off')
        
        # Plotar processada
        axes[i, 2].imshow(proc_img, cmap='gray')
        axes[i, 2].set_title(f'Processada\n{proc_img.shape}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Salvar figura
    output_dir = "../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'preprocessing_comparison.png'), dpi=150)
    plt.show()
    
    # Exibir estatísticas
    print("Estatísticas das imagens processadas:")
    print(f"Total de imagens originais: {len(original_images)}")
    
    processed_images = glob.glob(os.path.join(processed_dir, "**/*.jpg"), recursive=True)
    if not processed_images:
        processed_images = glob.glob(os.path.join(processed_dir, "**/*.png"), recursive=True)
    
    print(f"Total de imagens processadas: {len(processed_images)}")
    
    if processed_images:
        # Calcular estatísticas de tamanho
        sizes = []
        for img_path in processed_images[:100]:  # Amostra das 100 primeiras
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes.append(img.shape)
        
        if sizes:
            unique_sizes = set(sizes)
            print(f"\nTamanhos encontrados: {unique_sizes}")
            if len(unique_sizes) == 1:
                print("✓ Todas as imagens têm o mesmo tamanho!")
            else:
                print("✗ As imagens têm tamanhos diferentes!")

if __name__ == "__main__":
    # Configurar caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_dir = os.path.join(base_dir, "Images", "Selected_images")
    processed_dir = os.path.join(base_dir, "Images", "Optimized_images")
    
    visualize_preprocessing(original_dir, processed_dir, num_samples=5)