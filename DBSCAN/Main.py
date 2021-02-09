import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from data.Data import Data
from labels.Labels import Labels
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
"""
DBSCAN

Muitas das técnicas tradicionais de agrupamento, como k-médias, agrupamento hierárquico e fuzzy, podem ser usadas para agrupar dados sem supervisão.
No entanto, quando aplicadas a tarefas com clusters de forma arbitrária, ou clusters dentro de cluster, as técnicas tradicionais podem ser incapazes de alcançar bons resultados. Ou seja, os elementos no mesmo cluster podem não ter similaridade suficiente ou o desempenho pode ser ruim. Além disso, o clustering baseado em densidade localiza regiões de alta densidade que são separadas umas das outras por regiões de baixa densidade. A densidade, neste contexto, é definida como o número de pontos dentro de um raio especificado.
Nesta seção, o foco principal será manipular os dados e propriedades do DBSCAN e observar o agrupamento resultante.
"""
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#Constantes
nun_days = 15000                                  #numero de candles
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
X_trainset, X_testset, y_trainset, y_testset = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_trainset.shape,  y_trainset.shape)
print ('Test set:', X_testset.shape,  y_testset.shape)

"""
Modelo e Avaliação

Primeiro, criaremos uma instância do DecisionTreeClassifier chamada traderTree. Dentro do classificador, especifique criterion = "entropia" para que possamos ver o ganho de informação de cada nó.
"""
traderTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
traderTree.fit(X_trainset,y_trainset)
predTree = traderTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

"""
Visualização
Vamos visualizar a árvore
"""
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "trader1.png";
featureNames = data_labels.columns[0:19]
targetNames = ['compra','SO','venda'] #data_labels["target"].unique().tolist()
out=tree.export_graphviz(traderTree,feature_names=featureNames, out_file=dot_data, class_names= targetNames, filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
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








