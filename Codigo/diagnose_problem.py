# Codigo/diagnose_problem.py
import os
import numpy as np

def diagnose_problem():
    """Diagnostica o problema de identificacao de pessoas"""
    input_dir = "../Images/Selected_images"
    
    # Listar alguns arquivos
    files = os.listdir(input_dir)[:20]
    print("Primeiros 20 arquivos:")
    for f in files:
        print(f"  {f}")
    
    # No CelebA, os nomes sao numericos sequenciais
    # Exemplo: 000001.jpg, 000002.jpg, etc.
    # Mas isso e o ID da IMAGEM, nao da PESSOA
    
    print("\nPROBLEMA IDENTIFICADO:")
    print("Os arquivos tem nomes como '145328.jpg' - sao IDs de IMAGENS, nao de PESSOAS.")
    print("No CelebA, precisamos do arquivo identity_CelebA.txt para mapear imagem->pessoa.")
    
    print("\nSOLUCOES:")
    print("1. Se tiver identity_CelebA.txt, use-o para mapear corretamente")
    print("2. Se nao tiver, podemos criar um mapeamento artificial para teste")
    print("3. Ou usar outra estrategia para agrupar imagens por pessoa")

if __name__ == "__main__":
    diagnose_problem()