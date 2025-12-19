Sistema de Reconhecimento Facial - CelebA

Trabalho de SIN5016 - Aprendizado de Máquina - Atividade 1
Descrição do Projeto

Implementação de um sistema de reconhecimento facial utilizando a base de dados CelebA, empregando dois classificadores (MLP e SVM) com extração de características via descritores HOG e LBP, conforme especificações da atividade.
Objetivos

    Implementar 2 classificadores para tarefa de reconhecimento facial (autenticação e identificação)

    Extrair características usando descritor HOG (obrigatório) e LBP (opcional combinado com HOG)

    Empregar validação cruzada 5-fold para avaliação

    Limitar experimentos de identificação a 5.000 amostras conforme especificação

    Gerar modelos comparativos para análise de desempenho

Estrutura do Projeto
text

.
├── Codigo/                    # Código fonte implementado
│   ├── train_classifiers.py   # Script principal de treinamento
│   ├── extract_features.py    # Extração de características HOG/LBP
│   ├── preprocess_images.py   # Pré-processamento de imagens
│   └── analyze_features.py    # Análise de características
├── data/                      # Dados e características extraídas
│   ├── features/              # Features HOG, LBP e COMBINADO
│   ├── processed/             # Dados processados
│   └── raw/                   # Metadados originais CelebA
├── Execucao/                  # Modelos treinados e resultados (para entrega)
│   ├── Hog/                   # Resultados com HOG
│   └── Outro/                 # Resultados com COMBINADO (HOG+LBP)
├── results/                   # Resultados completos dos experimentos
│   ├── models/                # Modelos serializados
│   ├── logs/                  # Logs de execução
│   └── plots/                 # Gráficos e visualizações
├── Relatorio/                 # Relatórios e resumo dos resultados
├── Images/                    # Dataset de imagens CelebA
├── config/                    # Configurações do projeto
├── requirements.txt           # Dependências do projeto
└── README.md                  # Este arquivo

Base de Dados

CelebFaces Attributes (CelebA)

    202.599 imagens de celebridades

    10.177 identidades únicas

    40 atributos anotados por imagem

    Imagens de 218×178 pixels

    Usamos 20% da base (≈40.000 imagens) para experimentos

Tecnologias Utilizadas

    Python 3.11+

    scikit-learn: MLP, SVM, validação cruzada, PCA

    scikit-image: Extração de características HOG e LBP

    OpenCV: Pré-processamento de imagens

    numpy/pandas: Manipulação de dados

    matplotlib: Visualização de resultados

Especificações Técnicas

Classificadores Implementados

    MLP (Multilayer Perceptron)

        1 camada escondida (100 neurônios)

        Algoritmo backpropagation com Adam

        Critério de parada antecipada (patience=20)

    SVM (Support Vector Machine)

        Tipo C-SVC tradicional

        Kernel RBF

        Parâmetros padrão (C=1.0, gamma='scale')

Descritores de Características

    HOG (Histogram of Oriented Gradients)

        Células de 8×8 pixels

        Blocos de 2×2 células

        9 orientações

        Dimensão: 1764 features por imagem

    LBP (Local Binary Patterns)

        Combinado com HOG para formação do descritor COMBINADO

        Dimensão combinada: 1790 features por imagem

Metodologia de Avaliação

    Validação cruzada: 5-fold estratificado

    Balanceamento: Apenas classes com ≥5 amostras

    Métricas: Acurácia, Precisão, Recall, AUC (para verificação)

    PCA: Redução para 100 componentes (variância preservada >85%)

Como Executar
1. Configuração do Ambiente
bash

# Clonar repositório
git clone <repositorio>
cd sin5016-atividade1

# Ativar ambiente virtual (se existente)
source sin5016/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt

2. Extração de Características (se necessário)
bash

# Extrair características HOG e LBP
python Codigo/extract_features.py

# Verificar extração
python Codigo/analyze_features.py

3. Executar Experimentos Completos
bash

# Executar pipeline completo (verificação + identificação)
python Codigo/train_classifiers.py

4. Gerar Arquivos de Execução (para entrega)

Os arquivos de execução são gerados automaticamente em ../Execucao/ contendo:

    config.txt: Configurações dos parâmetros

    error.txt: Histórico de erro por época

    model.dat: Modelo serializado (placeholder)

Resultados Obtidos
Verificação (Autenticação) - 1:1 Matching
Descritor	Modelo	Acurácia CV (5-fold)	Acurácia Teste	AUC
HOG	MLP	65.78% ±0.47%	66.66%	66.66%
HOG	SVM	70.64% ±1.26%	72.26%	72.26%
HOG+LBP	MLP	66.36% ±0.93%	66.92%	66.92%
HOG+LBP	SVM	70.83% ±1.08%	72.58%	72.58%

Melhor resultado: SVM com HOG+LBP (72.58% acurácia)
Identificação - 1:N Matching
Nº Classes	Modelo	Acurácia Teste	Baseline	Ganho
10	SVM	60.00%	10.00%	+500%
20	SVM	48.96%	5.00%	+879%
30	SVM	47.16%	3.33%	+1315%
200	SVM	7.34%	0.50%	+1368%

Observações:

    SVM consistentemente superior ao MLP

    Performance decai com aumento de classes

    Identificação viável para grupos pequenos (≤30 pessoas)

Arquivos de Entrega
text

grupo_DD.zip/
├── Codigo/
│   ├── train_classifiers.py     # Script principal corrigido
│   ├── extract_features.py      # Extração de características
│   └── ... outros necessários
├── Execucao/                    # Estrutura completa conforme especificação
│   ├── Hog/
│   │   ├── Melhor/
│   │   │   ├── config.txt
│   │   │   ├── error.txt
│   │   │   └── model.dat
│   │   └── Pior/
│   └── Outro/
│       ├── Melhor/
│       └── Pior/
├── results/                     # Resultados completos
│   ├── experiment_summary_*.json
│   ├── experiment_summary_*.txt
│   ├── models/                  # Modelos treinados
│   └── plots/                   # Gráficos
├── Relatorio/                   # Resumo dos resultados
├── README.md                    # Este arquivo
└── requirements.txt             # Dependências

Análise Crítica
Pontos Fortes

    Sistema implementado conforme especificação

    SVM apresenta bom desempenho na verificação (~72%)

    Identificação funcional para grupos pequenos

    Validação cruzada robusta (5-fold estratificado)

Limitações

    MLP apresenta overfitting na identificação

    Features HOG têm poder discriminativo limitado para muitas classes

    Performance decai rapidamente acima de 30 classes

Sugestões de Melhoria

    Uso de deep features (CNN pré-treinada)

    Data augmentation para aumentar amostras

    Tuning mais agressivo de hiperparâmetros

    Ensemble de classificadores

Equipe

    Edson de Oliveira Vieira - 16294075

    Antonio

Referências

    Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. ICCV.

    Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.

    Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI.

    Material da disciplina SIN5016 - Aprendizado de Máquina, USP.

Licença

Este projeto é desenvolvido para fins acadêmicos na disciplina SIN5016 - Aprendizado de Máquina da EACH/USP.

Última atualização: Dezembro 2024
Disciplina: SIN5016 - Aprendizado de Máquina
Instituição: Escola de Artes, Ciências e Humanidades - Universidade de São Paulo (EACH/USP)
