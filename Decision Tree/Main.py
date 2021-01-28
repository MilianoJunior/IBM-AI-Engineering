
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from data.Data import Data
from labels.Labels import Labels

"""
K-Nearest Neighbors

K-Nearest Neighbours é um algoritmo para aprendizagem supervisionada. Onde os dados são 'treinados' com pontos de dados 
correspondentes à sua classificação. Uma vez que um ponto deve ser previsto, ele leva em consideração os 'K' pontos mais 
próximos dele para determinar sua classificação.

"""
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#Constantes
nun_days = 950                                   #numero de candles
batch_size = 1                                   #divisao em blocos
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#instanciar objetos
data = Data(nun_days,batch_size)
entrada,entrada_trader,base,media,std = data.import_data()
labels = Labels()

data_labels = labels.index_labels(base,entrada)
