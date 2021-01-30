# K-Means Clustering
Existem muitos modelos de cluster por aí. Neste notebook apresentaremos o modelo que é considerado um dos modelos mais simples entre eles. Apesar de sua simplicidade, o K-means é amplamente utilizado para armazenamento em cluster em muitos aplicativos de ciência de dados, especialmente útil se você precisar descobrir rapidamente insights de dados não rotulados. Neste bloco de notas, você aprenderá como usar k-Means para segmentação de operações na bolsa de valores.

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
A padronização de dados dá aos dados média zero e variação unitária, é uma boa prática.
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

```
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)
```
### Avaliação
Podemos verificar facilmente os valores calculando a acurácia comparando aos rótulos com segmentação.
```
data_labels["Clus_km"] = labels
data_labels.head(5)
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(data_labels['target'],data_labels['Clus_km']))
Train set Accuracy:  0.19658119658119658
```
### Visualização

![alt text](https://github.com/MilianoJunior/IBM-AI-Engineering/blob/master/K-Means%20Clustering/Figure%202021-01-30%20151746.png?raw=true)

![alt text](https://github.com/MilianoJunior/IBM-AI-Engineering/blob/master/K-Means%20Clustering/Figure%202021-01-30%20152000.png?raw=true)

## Conclusão

A segmentação de dados pode ser uma etapa importante na etapa de pré processamento, auxiliando na escolha e filtros de dados que não sejam úteis para um modelo previsor.






