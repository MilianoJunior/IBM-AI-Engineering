import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from data.Data import Data
from labels.Labels import Labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
"""
Hierarchical Clustering

Estaremos examinando uma técnica de agrupamento, que é o agrupamento hierárquico aglomerativo. Lembre-se de que aglomerativo é a abordagem de baixo para cima.
Neste laboratório, veremos o clustering aglomerativo, que é mais popular do que o clustering divisivo.
objetivo: Encontrar os indicadores que apresentam caracteristicas semelhantes.
"""
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#Constantes
nun_days = 910                                  #numero de candles
batch_size = 1                                   #divisao em blocos
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
Seleção dos recursos
"""
colunas = ['Hora', 'dif', 'retracao +', 'retracao -', 'RSI', 'M22M44', 'M22M66',
            'M66M44', 'ADX', 'ATR', 'Momentum', 'CCI', 'Bears', 'Bulls', 'Stock1',
            'Stock2', 'Wilians', 'Std', 'MFI']
featureset = data_labels[colunas] #.values.astype(float)

"""
Normalização dos dados

Agora podemos normalizar o conjunto de recursos. MinMaxScaler transforma recursos, dimensionando cada recurso para um determinado intervalo. É por padrão (0, 1). Ou seja, esse estimador dimensiona e traduz cada recurso individualmente de forma que fique entre zero e um.
"""

x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]

"""
Clustering usando Scipy
"""





# #separando os dados
# colunas = ['Hora', 'dif', 'retracao +', 'retracao -', 'RSI', 'M22M44', 'M22M66',
#             'M66M44', 'ADX', 'ATR', 'Momentum', 'CCI', 'Bears', 'Bulls', 'Stock1',
#             'Stock2', 'Wilians', 'Std', 'MFI']
# X = data_labels[colunas].values.astype(float)
# y= data_labels['target']
# #normalização dos dados
# X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
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
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)


# import scipy
# leng = X.shape[0]
# D = scipy.zeros([leng,leng])
# for i in range(leng):
#     for j in range(leng):
#         D[i,j] = scipy.spatial.distance.euclidean(X[i], X[j])
# print(D)


# import pylab
# import scipy.cluster.hierarchy
# Z = hierarchy.linkage(D, 'complete')

# from scipy.cluster.hierarchy import fcluster
# max_d = 3
# clusters = fcluster(Z, max_d, criterion='distance')

# from scipy.cluster.hierarchy import fcluster
# k = 5
# clusters = fcluster(Z, k, criterion='maxclust')
# clusters

# fig = pylab.figure(figsize=(18,50))
# def llf(id):
#     return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
# dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
"""
Modelo e Avaliação

Vamos construir nosso modelo usando LogisticRegression do pacote Scikit-learn. Esta função implementa regressão logística e pode usar diferentes otimizadores numéricos para encontrar parâmetros, incluindo ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. Você pode encontrar muitas informações sobre os prós e os contras desses otimizadores se pesquisar na Internet.
A versão de Regressão Logística em Scikit-learn, suporta regularização. A regularização é uma técnica usada para resolver o problema de overfitting em modelos de aprendizado de máquina. O parâmetro C indica o inverso da força de regularização que deve ser uma flutuação positiva. Valores menores especificam regularização mais forte. Agora vamos ajustar nosso modelo com o conjunto de trem:
"""

"""
Avaliação
"""


"""
Testando para diversos parametros
"""

    
"""
Matriz de Confusão
Outra maneira de examinar a precisão do classificador é examinar a matriz de confusão.
"""

"""
Conclusão

É possível verificar que o modelo converge para uma solução. É interressante notar que quando k=1,
em treinamento é atingido 100% das soluções, mas quando colocado a prova no conjunto de dados teste
não se tem o mesmo número de acertos, isso significa que o modelo está com overfiting. 
Quando K vai aumentando, a acurracia dos dados de treinamento vai dimunindo,no entando,a acurracia dos
dados de teste tem um leve aumento.
Talvez, filtrar as entradas na busca de outliers ou dimunuir o número de entradas verificando a correlação
possam trazer melhores resultados.


"""








