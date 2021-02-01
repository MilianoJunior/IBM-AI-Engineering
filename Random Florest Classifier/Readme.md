# Random forest classifier

Uma floresta aleatória é um metaestimador que ajusta vários classificadores de árvore de decisão em várias subamostras do conjunto de dados e usa a média para melhorar a precisão preditiva e o sobreajuste de controle.

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
```
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predTree = clf.predict(X_test)
pred_proba = clf.predict_proba(X_test)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
```
DecisionTrees's Accuracy:  0.5425531914893617
Visualização
Vamos visualizar a árvore
![alt text](https://github.com/MilianoJunior/IBM-AI-Engineering/blob/master/Decision%20Tree/trader.png?raw=true)
## Salvando o modelo
```
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
```
## Conclusão

É possível verificar que o modelo converge para uma solução, com acurácia de 58%.Na paste tests criado um rede socket para
comunicar com o metatrader 5, foram feitos backtests, o modelo fez algumas operações lucarativas, mas perdeu em outras, 
não obetendo resultado satisfatório para executar em conta real.

