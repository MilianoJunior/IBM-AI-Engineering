# Logistic regression

Regressão logística é uma técnica estatística e aprendizado de máquina para classificar registros. Para entender melhor, podemos apresentar o exemplo de negociações na bolsa de valores, os operadores realizam compra e venda de ativos com base no conhecimento da analise técnica, sendo assim, eles observam diversos indicadores e a variação do preço das ações, e decidem quando comprar, vender ou não operar, as melhores negociações geram lucros. Pode-se dizer que as variáveis independentes são indicadores e variação do preço, variáveis dependentes são comprar, vender ou não operar.
Na regressão logística, as variáveis independentes devem ser contínuas. No entanto, nota-se que a variável dependente que é objeto de previsão do modelo de regressão logística, deve ser categórica ou binaria. Sendo assim, a diferença da regressão logística para a regressão linear, está presente na variável dependente, no qual a regressão linear deva ser continua.
Para resolver os problemas de regressão logística, o algoritmo faz cálculos não lineares, a função sigmoide é um cálculo realizado que indiferente dos valores de entrada, sempre retorna valores probabilísticos entre 0 e 1.

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

Vamos construir nosso modelo usando LogisticRegression do pacote Scikit-learn. Esta função implementa regressão logística e pode usar diferentes otimizadores numéricos para encontrar parâmetros, incluindo ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. Você pode encontrar muitas informações sobre os prós e os contras desses otimizadores se pesquisar na Internet.
A versão de Regressão Logística em Scikit-learn, suporta regularização. A regularização é uma técnica usada para resolver o problema de overfitting em modelos de aprendizado de máquina. O parâmetro C indica o inverso da força de regularização que deve ser uma flutuação positiva. Valores menores especificam regularização mais forte. Agora vamos ajustar nosso modelo com o conjunto de trem:
```
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
```
### Avaliação
```
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
jaccard_score(y_test, yhat,pos_label=0)
```
| Type | precision | recall | f1-score | support |
| ------ | ------ |------ | ------ | ------ |
| 0.0 | 0.63 | 0.79 | 0.70 | 53 |
| 1.0 | 0.23 | 0.38 | 0.29 | 16 |
| 2.0 | 0.00 | 0.00 | 0.00 | 25 |


### Testando para diversos parametros
```
params = ['newton-cg', 'lbfgs', 'sag', 'saga']
for i in params:
    LR = LogisticRegression(C=0.01, solver=i).fit(X_train,y_train)
    yhat = LR.predict(X_test)
    print('----------')
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, LR.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

Train set Accuracy:  0.5561497326203209
Test set Accuracy:  0.5106382978723404
----------
Train set Accuracy:  0.5508021390374331
Test set Accuracy:  0.5319148936170213
----------
Train set Accuracy:  0.5508021390374331
Test set Accuracy:  0.5319148936170213
----------
Train set Accuracy:  0.5561497326203209
Test set Accuracy:  0.5106382978723404
----------
Train set Accuracy:  0.5508021390374331
Test set Accuracy:  0.5319148936170213
----------
Train set Accuracy:  0.5508021390374331
Test set Accuracy:  0.5319148936170213
```
### Matriz de Confusão
Outra maneira de examinar a precisão do classificador é examinar a matriz de confusão, para o parametro liblinear.

![alt text](https://github.com/MilianoJunior/IBM-AI-Engineering/blob/master/Regression%20Logistic/Figure%202021-01-30%20095647.png?raw=true)

## Conclusão

A matriz de confusão mostra a performace desta técnica, com uma acurácia de 51%, o modelo não apresenta uma solução para se obter negociações lucrativas.




