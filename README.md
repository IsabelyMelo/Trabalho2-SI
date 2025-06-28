# Projeto de Classificação Supervisionada - Trabalho 2
## Alunos:

- ### Caio Cezar Dias
- ### Isabely Toledo de Melo

## Descrição do Projeto

Este repositório apresenta a implementação manual de dois algoritmos de aprendizado supervisionado: **K-Nearest Neighbors (KNN)** e **Perceptron Linear**, aplicados em dois problemas clássicos de classificação:

- **Problema 1 - Azul/Laranja**: Classificação binária com base em duas características (`x1`, `x2`) a partir do dataset `dados_azul_laranja.csv`.
- **Problema 2 - Iris**: Classificação multiclasse para identificar espécies de flores (Setosa, Versicolor, Virginica) com base em atributos medidos.

O trabalho segue os critérios do enunciado, utilizando apenas implementações manuais (sem bibliotecas de machine learning), o que possibilita alcançar até **100% da nota**.

## Estrutura de Arquivos

```
├── Datasets/
│   └── dados_azul_laranja.csv
├── Exports/
│   ├── grafico_knn_azul_laranja.png
│   ├── grafico_perceptron_azul_laranja.png
│   ├── grafico_knn_iris.png
│   └── grafico_perceptron_iris.png
├── main.py
├── requirements.txt
└── README.md
```

- **main.py**: contém as funções manuais de divisão dos dados, implementação do KNN e Perceptron, além dos scripts para análise, treinamento, teste e visualização.
- **Datasets/**: pasta com o arquivo CSV fornecido para o problema 1.
- **Exports/**: pasta com os gráficos de saída gerados automaticamente.
- **README.md**: este arquivo com explicações sobre o projeto.

## Instruções de Execução

1. Certifique-se de ter o Python instalado (>= 3.8).
2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute o código:
   ```bash
   python main.py
   ```

## Breve Explicação das Escolhas

### Algoritmos Utilizados:
- **KNN (K-Nearest Neighbors)**:
  - Baseado na proximidade dos dados, com k=3.
  - Implementado manualmente.
  - Fronteiras de decisão não-lineares e robusto para poucos atributos.

- **Perceptron Linear**:
  - Algoritmo linear simples.
  - Estratégia One-vs-All para multiclasse.
  - Rápido, interpretável, mas limitado a separabilidade linear.

### Visualizações:
- Gráficos com fronteiras de decisão salvos na pasta `Exports/`.

### Divisão dos Dados:
- Divisão manual 80/20 com `seed=42` para reprodutibilidade.

## Considerações Finais

- Código sem uso de bibliotecas de machine learning para lógica dos modelos.
- Projeto reforça aprendizado prático e visual de classificação.