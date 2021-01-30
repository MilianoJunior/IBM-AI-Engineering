# Decision Trees

Neste algoritmo será implementado a árvore de decisão.Você usará este algoritmo de classificação para construir um modelo
a partir de dados históricos da bolsa de valores brasileira, no periodo intraday para o indice futuro. Em seguida, você usa 
a árvore de decisão treinada para prever a classe target, essa classe é estruturada para comprar e vender e não se posicionar 
em todos fechamentos de candle.

## Sobre os dados
Estes dados são informações retiradas da BMF Bovespa, o periodo é Intraday,além das informações que formam um candlestick, são associados as colunas, informações de indicadores técnicos.
```
colunas = (['Hora', 'dif', 'retracao +', 'retracao -', 'RSI', 'M22M44', 'M22M66',
           'M66M44', 'ADX', 'ATR', 'Momentum', 'CCI', 'Bears', 'Bulls', 'Stock1',
           'Stock2', 'Wilians', 'Std', 'MFI', 'target'],dtype='object')
```
A coluna target representa os rótulos,iformações que consideram a tendência do preços,onde:
  - 1 compra
  - 2 venda
  - 0 sem operação
```
data = Data(nun_days,batch_size)
entrada,entrada_trader,base,media,std = data.import_data()
labels = Labels()
data_labels = labels.index_labels(base,entrada)
```
## Normalização dos dados
A padronização de dados dá aos dados média zero e variação unitária, é uma boa prática,especialmente para algoritmos como KNN, que é baseado na distância dos casos:
```
X = data_labels[colunas].values.astype(float)
y= data_labels['target']
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
```
## Divisão dos em treinamento e teste
Fora da precisão da amostra é a porcentagem de previsões corretas que o modelo faz nos dados nos quais o modelo NÃO foi treinado. Fazer um treinamento e teste no mesmo conjunto de dados provavelmente terá baixa precisão fora da amostra, devido à probabilidade de ajuste excessivo.
É importante que nossos modelos tenham uma alta precisão fora da amostra, porque o objetivo de qualquer modelo, é claro, é fazer previsões corretas sobre dados desconhecidos. Então, como podemos melhorar a precisão fora da amostra? Uma maneira é usar uma abordagem de avaliação chamada Divisão de Treino / Teste. A divisão de treinamento / teste envolve a divisão do conjunto de dados em conjuntos de treinamento e teste, respectivamente, que são mutuamente exclusivos. Depois disso, você treina com o conjunto de treinamento e testa com o conjunto de teste.
```
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
```
## Modelo e Avaliação

Primeiro, criaremos uma instância do DecisionTreeClassifier chamada traderTree. Dentro do classificador, especifique criterion = "entropia" para que possamos ver o ganho de informação de cada nó.
```
traderTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
traderTree.fit(X_trainset,y_trainset)
predTree = traderTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
```
![alt text](https://github.com/MilianoJunior/IBM-AI-Engineering/blob/master/Decision%20Tree/trader.png?raw=true)
## Conclusão

Observando a arvore de decisão plotada com ajuda da biblioteca pydotplus, pode-se formular diversas regras para implementar um algoritmo trader. É impressionante está técnica com ajuda do calculo computacional,
No final do ramos, é verificar possiveis negociações que podem ser lucrativas.

