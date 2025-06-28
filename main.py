# Caio Cezar Dias e Isabely Toledo de Melo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import the os module for path manipulation

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris # Usado para carregamento e preparo dos dados

# --- Funções Auxiliares (Implementações Manuais) ---

def split_data(X, y, test_size=0.2, seed=42):
    """
    Divide manualmente os dados em conjuntos de treinamento e teste.
    
    Argumentos:
        X (np.array): Dados de características.
        y (np.array): Rótulos alvo.
        test_size (float): Proporção do conjunto de dados a ser incluída na divisão de teste.
        seed (int): Semente para geração de números aleatórios para garantir reprodutibilidade.
        
    Retorna:
        tupla: X_train, X_test, y_train, y_test
    """
    np.random.seed(seed) 
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    num_amostras_teste = int(test_size * len(X))
    idx_teste = indices[:num_amostras_teste]
    idx_treino = indices[num_amostras_teste:]

    X_train, X_test = X[idx_treino], X[idx_teste]
    y_train, y_test = y[idx_treino], y[idx_teste]
    
    return X_train, X_test, y_train, y_test

def knn_predict(X_train, y_train, X_test, k=3):
    """
    Implementa manualmente a previsão do K-Nearest Neighbors (KNN).
    
    Para cada ponto de teste, calcula sua distância a todos os pontos de treinamento,
    seleciona os k vizinhos mais próximos e atribui a classe com base na votação majoritária
    entre esses vizinhos.
    
    Argumentos:
        X_train (np.array): Características de treinamento.
        y_train (np.array): Rótulos de treinamento.
        X_test (np.array): Características de teste.
        k (int): Número de vizinhos mais próximos a considerar.
        
    Retorna:
        np.array: Rótulos previstos para o conjunto de teste.
    """
    predictions = []
    for ponto_teste in X_test:
        # Calcula a distância Euclidiana do ponto de teste para todos os pontos de treinamento
        distances = np.linalg.norm(X_train - ponto_teste, axis=1)
        
        # Obtém os índices dos k vizinhos mais próximos (os k menores distâncias)
        k_indices = distances.argsort()[:k]
        
        # Obtém os rótulos dos k vizinhos mais próximos
        k_labels = y_train[k_indices]
        
        # Determina a classe mais comum entre os k vizinhos (votação majoritária)
        # np.bincount conta as ocorrências de cada inteiro não negativo em um array
        # argmax retorna o índice do valor máximo, que corresponde à classe majoritária
        pred = np.bincount(k_labels).argmax()
        predictions.append(pred)
    return np.array(predictions)

def train_linear_classifier(X, y, num_classes=2, epochs=1000, lr=0.01):
    """
    Treina manualmente um classificador linear (Perceptron).
    
    Para classificação binária, treina um único Perceptron.
    Para classificação multi-classe (num_classes > 2), usa uma abordagem One-vs-All.
    
    Argumentos:
        X (np.array): Características de treinamento.
        y (np.array): Rótulos de treinamento.
        num_classes (int): Número de classes únicas na variável alvo.
        epochs (int): Número de iterações sobre os dados de treinamento.
        lr (float): Taxa de aprendizado, controla o tamanho do passo durante as atualizações de peso.
        
    Retorna:
        np.array: Pesos aprendidos para o(s) classificador(es) linear(es).
                  Para binário, a forma é (num_features + 1,).
                  Para multi-classe, a forma é (num_classes, num_features + 1).
    """
    # Adiciona um termo de viés (intercepto) às características adicionando uma coluna de uns
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  
    
    if num_classes == 2:  # Classificação Binária (Perceptron)
        weights = np.zeros(X_bias.shape[1])
        for _ in range(epochs):
            for i in range(X_bias.shape[0]):
                # Calcula a soma ponderada
                z = np.dot(X_bias[i], weights)
                # Função de ativação (função degrau para Perceptron)
                pred = 1 if z >= 0 else 0
                # Calcula o erro
                error = y[i] - pred
                # Atualiza os pesos
                weights += lr * error * X_bias[i]
    else:  # Classificação Multi-classe (Perceptron One-vs-All)
        weights = np.zeros((num_classes, X_bias.shape[1]))
        for _ in range(epochs):
            for i in range(X_bias.shape[0]):
                xi = X_bias[i]
                yi = y[i]
                # Calcula as pontuações para todos os classificadores
                scores = np.dot(weights, xi)
                # Prevê a classe com a pontuação mais alta
                pred = np.argmax(scores)
                # Atualiza os pesos apenas se a previsão estiver incorreta
                if pred != yi:
                    weights[yi] += lr * xi   # Aumenta o peso para a classe correta
                    weights[pred] -= lr * xi # Diminui o peso para a classe incorretamente prevista
    return weights

def predict_linear(X, weights):
    """
    Prevê manualmente os rótulos usando um classificador linear treinado.
    
    Argumentos:
        X (np.array): Características para previsão.
        weights (np.array): Pesos aprendidos da fase de treinamento.
        
    Retorna:
        np.array: Rótulos previstos.
    """
    # Adiciona um termo de viés às características
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    if weights.ndim == 1:  # Classificação Binária
        # Se a soma ponderada for >= 0, prevê 1, caso contrário 0
        return (np.dot(X_bias, weights) >= 0).astype(int)
    else:  # Classificação Multi-classe (One-vs-All)
        # Calcula as pontuações para cada classe e prevê a classe com a pontuação mais alta
        return np.argmax(np.dot(X_bias, weights.T), axis=1)

def plot_decision_boundary(model_predict_func, X, y, title, output_filename, model_params=None, labels_map=None, plot_features=None): # Added output_filename
    """
    Plota a fronteira de decisão de um classificador e salva o gráfico.
    
    Esta função cria uma grade de pontos no espaço de características e usa o
    modelo treinado para prever a classe para cada ponto, visualizando assim as
    regiões de decisão.
    
    Argumentos:
        model_predict_func (função): A função de previsão do modelo treinado.
                                       Deve aceitar (X, model_params) para linear
                                       ou (X_train, y_train, X_grid) para KNN.
        X (np.array): Dados de características usados para plotar os pontos de dispersão e definir os limites da grade.
        y (np.array): Rótulos verdadeiros correspondentes a X.
        title (str): Título do gráfico.
        output_filename (str): Nome do arquivo para salvar o gráfico.
        model_params (np.array, opcional): Parâmetros (ex: pesos) exigidos por model_predict_func.
        labels_map (dict, opcional): Um dicionário que mapeia rótulos numéricos de volta para nomes originais
                                     para a legenda do gráfico.
        plot_features (list, opcional): Nomes das características sendo plotadas, para os rótulos dos eixos.
    """
    # Define o tamanho do passo para a grade
    h = 0.01
    
    # Determina os valores mínimos e máximos para cada característica para definir os limites da grade
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Cria uma grade de pontos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Achata os pontos da grade para criar entrada para a função de previsão
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Faz previsões nos pontos da grade
    if model_params is not None:
        Z = model_predict_func(grid_points, model_params)
    else:
        # Para KNN, a função de previsão precisa de X_train e y_train para contexto
        # Passamos os X_train e y_train originais relevantes para o problema sendo plotado
        if 'Azul/Laranja' in title: 
            Z = model_predict_func(X_train_global_azul_laranja, y_train_global_azul_laranja, grid_points)
        elif 'Iris' in title:
             Z = model_predict_func(X_train_global_iris_2d, y_train_global_iris_2d, grid_points)
        else:
            raise ValueError("Não foi possível determinar os dados de treinamento apropriados para o gráfico KNN.")

    # Redimensiona as previsões de volta para a forma da grade
    Z = Z.reshape(xx.shape)

    # Define mapas de cores para as regiões de decisão e pontos de dispersão
    if labels_map and len(labels_map) == 2: # Para classificação binária
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    elif labels_map and len(labels_map) == 3: # Para Iris (3 classes)
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00AA00'])
    else: # Padrão para número desconhecido de classes
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFFFAA', '#AAFFFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00AA00', '#AAAA00', '#00AAAA'])

    plt.figure(figsize=(7, 6))
    # Plota as regiões de decisão
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    
    # Plota os pontos de treinamento
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50, alpha=0.9)
    
    plt.title(title, fontsize=14)
    if plot_features:
        plt.xlabel(plot_features[0], fontsize=12)
        plt.ylabel(plot_features[1], fontsize=12)
    else:
        plt.xlabel("Característica 1", fontsize=12)
        plt.ylabel("Característica 2", fontsize=12)
    
    # Cria uma legenda
    if labels_map:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                              markerfacecolor=cmap_bold(class_idx / (len(labels_map) - 1)), 
                              markersize=10, markeredgecolor='k') 
                   for label, class_idx in labels_map.items()]
        plt.legend(handles=handles, title="Classes", loc="upper left", bbox_to_anchor=(1, 1))

    plt.grid(True)
    plt.tight_layout() # Ajusta o layout para evitar que os rótulos se sobreponham
    
    # Save the plot to the 'Exports' directory
    exports_dir = 'Exports'
    if not os.path.exists(exports_dir):
        os.makedirs(exports_dir)
    plt.savefig(os.path.join(exports_dir, output_filename))
    plt.close() # Close the plot to free memory
    print(f"Gráfico '{output_filename}' salvo em '{exports_dir}/'")


# --- Variáveis globais para a função de plotagem do KNN (gambiarra, mas necessária para o plot manual da fronteira KNN) ---
# Estas variáveis são usadas pela função plot_decision_boundary quando 'model_params' não é fornecido
# (caso do KNN, que precisa de X_train e y_train para prever pontos na grade).
X_train_global_azul_laranja = None
y_train_global_azul_laranja = None
X_train_global_iris_2d = None
y_train_global_iris_2d = None

# --- Execução Principal ---

def run_azul_laranja_classification():
    """
    Lida com o problema de classificação para 'dados_azul_laranja.csv'.
    """
    print("--- Executando Classificação Azul/Laranja ---")

    # 1. Carregamento dos Dados
    # O conjunto de dados contém duas características de entrada (x1, x2) e uma classe categórica (Azul, Laranja).
    # Updated path to load from 'Datasets'
    data_path = os.path.join('Datasets', 'dados_azul_laranja.csv')
    df = pd.read_csv(data_path)  

    print("Amostras dos dados (dados_azul_laranja.csv):\n", df.head())

    # 2. Pré-processamento dos Dados
    # Extrai as características (X) e os rótulos alvo originais (y_original).
    X = df[['x1', 'x2']].values
    y_original = df['Classe'].values

    # 3. Conversão de 'azul' e 'laranja' em rótulos numéricos (0 e 1)
    # Isso é necessário para cálculos numéricos em algoritmos de classificação.
    labels_map_azul_laranja = {'azul': 0, 'laranja': 1}
    y = np.array([labels_map_azul_laranja[val.lower()] for val in y_original])

    # 4. Divisão Manual entre Treino e Teste
    # Uma semente fixa (42) é usada para garantir a reprodutibilidade da divisão aleatória.
    global X_train_global_azul_laranja, y_train_global_azul_laranja
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)
    X_train_global_azul_laranja = X_train # Armazena para o plot da fronteira KNN
    y_train_global_azul_laranja = y_train # Armazena para o plot da fronteira KNN

    # 5. Implementação Manual do KNN (K-Nearest Neighbors)
    # Usando k=3 como um valor comum e eficaz para tarefas de classificação simples.
    print("\n--- KNN (Manual) para Azul/Laranja ---")
    y_pred_knn = knn_predict(X_train, y_train, X_test, k=3)
    acc_knn = np.mean(y_pred_knn == y_test)
    print(f"Acurácia KNN (manual): {acc_knn:.2%}") # Verifica a taxa de acerto

    # 6. Implementação Manual do Classificador Linear (Perceptron Simples)
    # O algoritmo Perceptron atualiza iterativamente os pesos com base em erros de classificação.
    print("\n--- Classificador Linear (Perceptron Manual) para Azul/Laranja ---")
    weights_linear = train_linear_classifier(X_train, y_train, num_classes=2)
    y_pred_linear = predict_linear(X_test, weights_linear)
    acc_linear = np.mean(y_pred_linear == y_test)
    print(f"Acurácia Classificador Linear (manual): {acc_linear:.2%}") # Verifica a taxa de acerto

    # 7. Visualização das Fronteiras de Decisão
    # Esses gráficos ilustram como cada classificador separa os pontos de dados.
    print("\n--- Plotando Fronteiras de Decisão para Azul/Laranja ---")
    # Plot da Fronteira de Decisão do KNN
    plot_decision_boundary(knn_predict, X, y, 
                           "Fronteira de Decisão - KNN (Manual) - Azul/Laranja",
                           "knn_azul_laranja_boundary.png", # Output filename
                           labels_map={'azul': 0, 'laranja': 1},
                           plot_features=['x1', 'x2'])

    # Plot da Fronteira de Decisão do Classificador Linear
    plot_decision_boundary(predict_linear, X, y, 
                           "Fronteira de Decisão - Linear (Manual) - Azul/Laranja", 
                           "linear_azul_laranja_boundary.png", # Output filename
                           model_params=weights_linear,
                           labels_map={'azul': 0, 'laranja': 1},
                           plot_features=['x1', 'x2'])

    print("\n--- Classificação Azul/Laranja Concluída ---")


def run_iris_classification():
    """
    Lida com o problema de classificação para o conjunto de dados Iris.
    """
    print("\n--- Executando Classificação Iris ---")

    # 1. Carregamento e Preparação dos Dados
    # The Iris dataset is loaded via scikit-learn's `load_iris()`, so its path doesn't need to be changed.
    iris_bunch = load_iris()
    X = iris_bunch.data # Características: comprimento da sépala, largura da sépala, comprimento da pétala, largura da pétala
    y = iris_bunch.target # Alvo: espécies (0, 1, 2 para setosa, versicolor, virginica)
    feature_names = iris_bunch.feature_names
    target_names = iris_bunch.target_names

    # Converte para DataFrame para facilitar a análise exploratória
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['species'] = pd.Categorical.from_codes(y, target_names)

    # 2. Análise Exploratória de Dados (EDA)
    # Compreendendo a distribuição dos dados, tipos e possíveis valores ausentes.
    print("\nPrimeiros dados (Iris):\n", iris_df.head())
    print("\nÚltimos dados (Iris):\n", iris_df.tail())
    print("\nInformações do DataFrame (Iris):\n")
    iris_df.info()
    print("\nValores ausentes (Iris):\n", iris_df.isnull().sum())
    print("\nResumo estatístico (Iris):\n", iris_df.describe())
    print("\nDistribuição das espécies (Iris):\n", iris_df['species'].value_counts())

    # 3. Divisão Manual entre Treino e Teste
    # Dividindo o conjunto de dados Iris completo de 4 características para treinamento e teste.
    # Os dados são divididos em amostras de treinamento (X_train e y_train) e amostras de teste (X_test e y_test).
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)

    # 4. Implementação Manual do KNN (K-Nearest Neighbors)
    # Aplicando KNN ao conjunto de dados completo de 4 características.
    print("\n--- KNN (Manual) para Iris (Todas as Características) ---")
    y_pred_knn = knn_predict(X_train, y_train, X_test, k=3)
    acc_knn = np.mean(y_pred_knn == y_test)
    print(f"Acurácia KNN (manual): {acc_knn:.2%}") # Verifica a taxa de acerto

    # 5. Implementação Manual do Classificador Linear (Perceptron One-vs-All)
    # Usando um Perceptron multi-classe (estratégia One-vs-All) para o conjunto de dados Iris.
    print("\n--- Classificador Linear (Perceptron One-vs-All Manual) para Iris (Todas as Características) ---")
    weights_linear_iris = train_linear_classifier(X_train, y_train, num_classes=len(np.unique(y)))
    y_pred_linear = predict_linear(X_test, weights_linear_iris)
    acc_linear = np.mean(y_pred_linear == y_test)
    print(f"Acurácia Linear (manual): {acc_linear:.2%}") # Verifica a taxa de acerto

    # 6. Visualização das Fronteiras de Decisão (usando 2 atributos)
    # Para visualizar, reduzimos a dimensionalidade para as duas características da pétala.
    print("\n--- Plotando Fronteiras de Decisão para Iris (2 Características) ---")
    X_2d = X[:, [2, 3]] # Selecionando Comprimento da Pétala (índice 2) e Largura da Pétala (índice 3)
    
    global X_train_global_iris_2d, y_train_global_iris_2d
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = split_data(X_2d, y, test_size=0.2, seed=42)
    X_train_global_iris_2d = X_train_2d # Armazena para o plot da fronteira KNN
    y_train_global_iris_2d = y_train_2d # Armazena para o plot da fronteira KNN

    labels_map_iris = {name: i for i, name in enumerate(target_names)}

    # Treina novamente o KNN para dados 2D para plotagem da fronteira
    # Nota: knn_predict é genérico o suficiente para funcionar com dados 2D.
    plot_decision_boundary(knn_predict, X_2d, y, 
                           "Fronteira de Decisão - KNN (Manual) - Iris (Pétala)",
                           "knn_iris_petal_boundary.png", # Output filename
                           labels_map=labels_map_iris,
                           plot_features=['Comprimento da Pétala', 'Largura da Pétala'])

    # Treina novamente o Classificador Linear para dados 2D para plotagem da fronteira
    weights_linear_iris_2d = train_linear_classifier(X_train_2d, y_train_2d, num_classes=len(np.unique(y)))
    plot_decision_boundary(predict_linear, X_2d, y, 
                           "Fronteira de Decisão - Linear (Manual) - Iris (Pétala)", 
                           "linear_iris_petal_boundary.png", # Output filename
                           model_params=weights_linear_iris_2d,
                           labels_map=labels_map_iris,
                           plot_features=['Comprimento da Pétala', 'Largura da Pétala'])

    print("\n--- Classificação Iris Concluída ---")


if __name__ == "__main__":
    # --- Executa Problemas de Classificação ---
    run_azul_laranja_classification()
    print("\n" + "="*80 + "\n") # Separador para clareza
    run_iris_classification()