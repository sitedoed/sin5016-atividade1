# Sistema de Reconhecimento Facial - CelebA

Trabalho de SIN5016 - Classificação de Imagens com Descritores HOG e LBP
## Descrição do Projeto

Implementação de um sistema de reconhecimento facial utilizando a base de dados CelebA, empregando dois classificadores (MLP e SVM) com extração de características via descritores HOG e LBP.

## Objetivos

    Implementar 2 classificadores para tarefa de reconhecimento facial

    Extrair características usando descritor HOG (obrigatório) e LBP (opcional)

    Empregar validação cruzada k-fold (k=5) para avaliação

    Garantir balanceamento ≥30% das instâncias minoritárias

    Gerar modelos comparativos para análise de desempenho

## Estrutura do Projeto
text

.
├── Codigo/                    # Código fonte implementado
├── dados/                     # Metadados e características extraídas
├── Execucao/                  # Modelos treinados e resultados
├── images/                    # Dataset de imagens CelebA
├── Relatorio/                 # Relatórios e documentação
├── sin5016/                   # Ambiente virtual Python
├── experiments/               # Experimentos e checkpoints
├── requirements.txt           # Dependências do projeto
├── setup.sh                   # Script de configuração
└── README.md                  # Este arquivo

### Base de Dados

CelebFaces Attributes (CelebA)

    202.599 imagens de celebridades

    10.177 identidades únicas

    40 atributos anotados por imagem

    Imagens de 178×218 pixels

### Tecnologias Utilizadas

    Python 3.0+

    scikit-learn: MLP, SVM, validação cruzada

    scikit-image: Extração de características HOG e LBP

    OpenCV/PIL: Pré-processamento de imagens

    imbalanced-learn: Balanceamento de dados

    pandas/numpy: Manipulação de dados

### Especificações Técnicas
Classificadores Implementados

    MLP (Multilayer Perceptron)

        1 camada escondida

        Algoritmo backpropagation

        Critério de parada antecipada

    SVM (Support Vector Machine)

        Tipo C-SVC

        Kernel linear/RBF

Descritores de Características

    HOG (Histogram of Oriented Gradients)

        Células de 8×8 pixels

        Blocos de 2×2 células

        9 orientações

    LBP (Local Binary Patterns)

        Padrões uniformes

        8 pontos de vizinhança

        Raio 1

Metodologia de Avaliação

    Validação cruzada: k=5 folds

    Balanceamento: ≥30% instâncias minoritárias

    Métricas: Acurácia, Precisão, Recall, F1-Score

### Como Executar
1. Configuração do Ambiente
bash

### Clonar repositório
git clone <repositorio>
cd sin5016-atividade1

# Ativar ambiente virtual
source sin5016/bin/activate  # Linux/Mac
# ou
.\sin5016\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

2. Pré-processamento
bash

# Extrair características HOG
python Codigo/preprocessing/extrair_hog.py

# Extrair características LBP
python Codigo/preprocessing/extrair_lbp.py

3. Treinamento dos Modelos
bash

# Treinar todos os modelos
python Codigo/main.py --descritor hog --modelo mlp
python Codigo/main.py --descritor hog --modelo svm
python Codigo/main.py --descritor lbp --modelo mlp
python Codigo/main.py --descritor lbp --modelo svm

4. Gerar Resultados
bash

# Executar pipeline completo
python Codigo/pipeline_completo.py

📁 Estrutura de Saída (Execucao/)
text

Execucao/
├── Hog/
│   ├── Melhor/          # Melhor modelo com HOG
│   │   ├── config.txt   # Configurações do modelo
│   │   ├── error.txt    # Histórico de treinamento
│   │   └── model.dat    # Modelo serializado
│   └── Pior/           # Pior modelo com HOG
└── Outro/              # LBP ou outro descritor
    ├── Melhor/         # Melhor modelo com LBP
    └── Pior/           # Pior modelo com LBP

📊 Resultados Esperados
Modelo	Descritor	Acurácia (média)	Precisão	Recall	F1-Score
MLP	HOG	-	-	-	-
SVM	HOG	-	-	-	-
MLP	LBP	-	-	-	-
SVM	LBP	-	-		


👥 Equipe

    Antonio - Matrícula

    Edson de Oliveira Vieira - 16294075


📚 Referências

    Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. Proceedings of International Conference on Computer Vision (ICCV).

    Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.

    Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI.

📄 Licença

Este projeto é desenvolvido para fins acadêmicos na disciplina SIN5016.

Última atualização: Dezembro 2025
Disciplina: SIN5016 - Classificação de Imagens
Instituição: EACH/USP
