import numpy as np
import chardet
import pandas as pd
from comunica import  Comunica
from sklearn import preprocessing
import pickle



HOST = ''    # Host
PORT = 8888  # Porta
R = Comunica(HOST,PORT)
s = R.createServer()
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
while True:
    p,addr = R.runServer(s)
    jm = np.array([p[1:20]])
    # X = preprocessing.StandardScaler().fit(jm).transform(jm.astype(float))
    previsao2 = loaded_model.predict(jm)[0]
    print('recebido: ',jm)
    # print('X: ', X)
    print('previsao: ',previsao2)
    # print('----------------')
    d3 = loaded_model.predict_proba(jm)[0]
    d4 = d3[np.argmax(d3)]
    print('probabilidade: ',d3)
    if previsao2 == 0 and d4 >0.5:
        print(' ')
        print('Sem operacao')
        print(' ')
    if previsao2 == 1 and d4 >0.5:
        flag = "compra-{}".format(d3)
        # flag ="compra"
        print(' ')
        print('compra: ',previsao2)
        print(' ')
        R.enviaDados(flag,s,addr)
    if previsao2 == 2 and d4 >0.5:
        flag = "venda-{}".format(d3)
        # flag = "venda"
        print(' ')
        print('venda: ',previsao2)
        print(' ')
        R.enviaDados(flag,s,addr)


