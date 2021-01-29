# K-Nearest Neighbors

K-Nearest Neighbours é um algoritmo para aprendizagem supervisionada. Onde os dados são 'treinados' com pontos de dados correspondentes à sua classificação. Uma vez que um ponto deve ser previsto, ele leva em consideração os 'K' pontos mais próximos dele para determinar sua classificação.

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
## Classificador
Classificador que implementa a votação de k-vizinhos mais próximos.
```
k = 4              # fator que define a abrangência dos vizinhos e sua correlação
test = []
train =[]
for K in range(1,50):
    neigh = KNeighborsClassifier(n_neighbors = K).fit(X_train,y_train)
    score_train = metrics.accuracy_score(y_train, neigh.predict(X_train))
    score_test = metrics.accuracy_score(y_test, neigh.predict(X_test))
    train.append(score_train)
    test.append(score_test)
```
The best accuracy was with 1.0 with k= 0
The best accuracy was with 0.5851063829787234 with k= 8
## Conclusão

É possível verificar que o modelo converge para uma solução. É interressante notar que quando k=1, em treinamento é atingido 100% das soluções, mas quando colocado a prova no conjunto de dados teste não se tem o mesmo número de acertos, isso significa que o modelo está com overfiting. 
Quando K vai aumentando, a acurracia dos dados de treinamento vai dimunindo,no entando,a acurracia dos dados de teste tem um leve aumento. Talvez, filtrar as entradas na busca de outliers ou dimunuir o número de entradas verificando a correlação possam trazer melhores resultados.




