import matplotlib.pyplot as plt
from sklearn import preprocessing
from data.Data import Data
from labels.Labels import Labels
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
"""
Random forest classifier

Uma floresta aleatória é um metaestimador que ajusta vários classificadores de árvore de decisão em várias subamostras do conjunto de dados e usa a média para melhorar a precisão preditiva e o sobreajuste de controle.
"""
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#Constantes
nun_days = 910                                #numero de candles
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
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

"""
Modelo e Avaliação
"""
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predTree = clf.predict(X_test)
pred_proba = clf.predict_proba(X_test)
print(pred_proba[0:5])
print (predTree [0:5])
print (y_test [0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

"""
Visualização
Vamos visualizar a árvore
"""
from sklearn import tree
from dtreeviz.trees import dtreeviz # will be used for tree visualization
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

"""
Salvando o modelo
"""
import pickle
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

plt.figure(figsize=(20,20))
_ = tree.plot_tree(clf.estimators_[0], feature_names=data_labels.columns, filled=True)

"""
Conclusão

É possível verificar que o modelo converge para uma solução, com acurácia de 58%.Na paste tests criado um rede socket para
comunicar com o metatrader 5, foram feitos backtests, o modelo fez algumas operações lucarativas, mas perdeu em outras, 
não obetendo resultado satisfatório para executar em conta real.

"""








