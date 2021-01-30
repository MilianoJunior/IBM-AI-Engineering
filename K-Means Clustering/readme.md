# Support Vector Machines

O SVM funciona mapeando dados para um espaço de recurso de alta dimensão para que os pontos de dados possam ser categorizados, mesmo quando os dados não são linearmente separáveis. Um separador entre as categorias é encontrado e os dados são transformados de forma que o separador possa ser desenhado como um hiperplano. Em seguida, as características dos novos dados podem ser usadas para prever o grupo ao qual um novo registro deve pertencer.

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
O algoritmo SVM oferece uma escolha de funções de kernel para realizar seu processamento. Basicamente, o mapeamento de dados em um espaço dimensional superior é chamado de kernelling. A função matemática usada para a transformação é conhecida como função kernel e pode ser de diferentes tipos, como:

* Linear
* Polinômio
* Função de base radial (RBF)
* Sigmóide
```
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
```
### Avaliação
```
yhat = clf.predict(X_test)
from sklearn.metrics import f1_score
print('Score: ',f1_score(y_test, yhat, average='weighted'))
Score:  0.488688367238291 
```
| Type | precision | recall | f1-score | support |
| ------ | ------ |------ | ------ | ------ |
| 0.0 | 0.65 | 0.85 | 0.74 | 53 |
| 1.0 | 0.27 | 0.38 | 0.32 | 16 |
| 2.0 | 0.33 | 0.04 | 0.07 | 25 |


### Testando para diversos parametros
```
for i in params:
    clf1 = svm.SVC(kernel=i)
    clf1.fit(X_train, y_train) 
    print('----------')
    print('Score: ',f1_score(y_test, yhat, average='weighted'),' Kernel: {}'.format(i)) 


Score:  0.488688367238291  Kernellinear
----------
Score:  0.488688367238291  Kernelpoly
----------
Score:  0.488688367238291  Kernelsigmoid
```
### Matriz de Confusão
Outra maneira de examinar a precisão do classificador é examinar a matriz de confusão, para o parametro rbf.

![alt text](https://github.com/MilianoJunior/IBM-AI-Engineering/blob/master/SVM/Figure%202021-01-30%20112932.png?raw=true)

## Conclusão

A matriz de confusão mostra a performace desta técnica, com uma acurácia de 51%, o modelo não apresenta uma solução para se obter negociações lucrativas.




