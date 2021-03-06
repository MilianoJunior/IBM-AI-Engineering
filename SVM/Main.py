import random 
import numpy as np
import matplotlib.pyplot as plt
from data.Data import Data
from labels.Labels import Labels
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
"""
K-Means Clustering
Existem muitos modelos de cluster por aí. Neste notebook apresentaremos o modelo que é considerado um dos modelos mais simples entre eles. Apesar de sua simplicidade, o K-means é amplamente utilizado para armazenamento em cluster em muitos aplicativos de ciência de dados, especialmente útil se você precisar descobrir rapidamente insights de dados não rotulados. Neste bloco de notas, você aprenderá como usar k-Means para segmentação de operações na bolsa de valores.
"""
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#Constantes
nun_days = 910                                  #numero de candles
batch_size = 1                                   #divisao em blocos
np.random.seed(0)
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#instanciar objetos
"""
Sobre os dados

Estes dados são informações retiradas da BMF Bovespa, o periodo é Intraday,além das informações que formam um candlestick, 
são associados as colunas, informações de indicadores técnicos.
Index(['Hora', 'dif', 'retracao +', 'retracao -', 'RSI', 'M22M44', 'M22M66',
       'M66M44', 'ADX', 'ATR', 'Momentum', 'CCI', 'Bears', 'Bulls', 'Stock1',
       'Stock2', 'Wilians', 'Std', 'MFI', 'target'],
      dtype='object')
O rótulos são iformações que consideram a tendência do preços, 1: compra, 2: venda e 0:sem operação
"""
data = Data(nun_days,batch_size)
entrada,entrada_trader,base,media,std = data.import_data()
labels = Labels()
data_labels = labels.index_labels(base,entrada)
print('Nome das colunas: ',data_labels.columns)
print('Quantidade de cada categória: ',data_labels.target.value_counts())
"""
Normalização dos dados

A padronização de dados dá aos dados média zero e variação unitária, é uma boa prática,
especialmente para algoritmos como KNN, que é baseado na distância dos casos:
"""
#separando os dados
colunas = ['Hora', 'dif', 'retracao +', 'retracao -', 'RSI', 'M22M44', 'M22M66',
            'M66M44', 'ADX', 'ATR', 'Momentum', 'CCI', 'Bears', 'Bulls', 'Stock1',
            'Stock2', 'Wilians', 'Std', 'MFI']
X = data_labels[colunas].values.astype(float)
y= data_labels['target']
#normalização dos dados
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
"""
Divisão dos em treinamento e teste

Fora da precisão da amostra é a porcentagem de previsões corretas que o modelo 
faz nos dados nos quais o modelo NÃO foi treinado. Fazer um treinamento e teste no
mesmo conjunto de dados provavelmente terá baixa precisão fora da amostra, devido à 
probabilidade de ajuste excessivo.
É importante que nossos modelos tenham uma alta precisão fora da amostra, porque o objetivo 
de qualquer modelo, é claro, é fazer previsões corretas sobre dados desconhecidos. 
Então, como podemos melhorar a precisão fora da amostra? Uma maneira é usar uma abordagem 
de avaliação chamada Divisão de Treino / Teste. A divisão de treinamento / teste envolve 
a divisão do conjunto de dados em conjuntos de treinamento e teste, respectivamente,
que são mutuamente exclusivos. Depois disso, você treina com o conjunto de treinamento 
e testa com o conjunto de teste.

"""
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

"""
Modelo e Avaliação

"""
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

"""
Avaliação
Podemos verificar facilmente os valores do centroide calculando a média dos recursos em cada cluster.
"""
data_labels["Clus_km"] = labels
data_labels.head(5)
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(data_labels['target'],data_labels['Clus_km']))

" Visualização"

data_labels.groupby('Clus_km').mean()

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 5], X[:, 6], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Hora', fontsize=18)
plt.ylabel('Dif', fontsize=16)
plt.show()


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Dif')
ax.set_ylabel('M22M44')
ax.set_zlabel('M22M66')

ax.scatter(X[:, 1], X[:, 5], X[:, 6], c= labels.astype(np.float))
