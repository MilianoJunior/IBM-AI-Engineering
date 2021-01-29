import pandas as pd
class Labels():
    def __init__(self,name='labels'):
        self.name = name
        self.valor_compra = 0
        self.valor_venda = 0
        self.compra = False
        self.venda = False
        self.SO = True
        self.cont_compra = 0
        self.cont_venda = 0
        self.tag = 35
    def taxa_de_retorno(self,x_inicial,x_final):
        return ((x_final-x_inicial)/x_inicial)*100
    def index_labels(self,base,input_rnn):
        dados = pd.DataFrame(columns=['Hora','dif','retracao +','retracao -',
                              'RSI', 'M22M44', 'M22M66', 'M66M44','ADX', 'ATR',
                              'Momentum', 'CCI', 'Bears', 'Bulls', 'Stock1',
                              'Stock2', 'Wilians', 'Std', 'MFI','target'])
        soma_dif = 0
        soma_high = 0
        soma_low = 0
        soma_compra = 0
        soma_venda = 0
        base.dif = base.dif.astype(float)
        base.open = base.open.astype(float)
        base.close = base.close.astype(float)
        base.high = base.high.astype(float)
        base.low = base.low.astype(float)
        for i in range(0,len(base)):
            soma_dif = soma_dif + base.dif[i]
            tr= self.taxa_de_retorno(base.open[i],base.close[i])
            soma_high =  base.high[i] - base.open[i]
            soma_low = base.low[i] - base.open[i]
            if base.open[i] < base.close[i]:
                tipo = 'alta'
        
            if base.open[i] == base.close[i]:
                tipo = 'igual'
                
            if base.open[i] > base.close[i]:
                tipo = 'baixa'
            if tipo == 'baixa' and base.dif[i] < -self.tag:
                if not self.venda:
                    self.compra = False
                    self.venda = True
                    self.SO= False
                    self.valor_venda = self.valor_venda + base.dif[i]
                    self.valor_compra = 0
                    self.cont_venda +=1
                else:
                    self.valor_venda = self.valor_venda + base.dif[i]
                operacao = 'venda'
                target =2
            if tipo == 'alta' and base.dif[i] > self.tag:
                if not self.compra:
                    self.compra = True
                    self.venda = False
                    self.SO = False
                    self.valor_compra = self.valor_compra + base.dif[i]
                    self.valor_venda = 0
                    self.cont_compra += 1
                else:
                    self.valor_compra = self.valor_compra + base.dif[i]
                operacao = 'compra'
                target =1
            if tipo == 'igual' or (base.dif[i] > -self.tag and base.dif[i] < self.tag):
                if self.compra:
                    if base.dif[i+1]>0 and base.dif[i] > -self.tag:
                        self.valor_compra = self.valor_compra + base.dif[i] 
                        operacao = 'compra'
                        target = 1
                    else:
                        if not self.SO:
                            self.compra = False
                            self.venda = False
                            self.SO = True
                            self.valor_compra = 0
                            self.valor_venda = 0
                            operacao = 'SO'
                            target = 0
                        
                if self.venda:
                    if base.dif[i+1]<0 and base.dif[i] < self.tag:
                        self.valor_venda = self.valor_venda + base.dif[i]
                        operacao = 'venda'
                        target = 1
                    else:
                        if not self.SO:
                            self.compra = False
                            self.venda = False
                            self.SO = True
                            self.valor_compra = 0
                            self.valor_venda = 0
                            operacao = 'SO'
                            target = 0
        
            if i >= 2:
                if base.dif[i-1]>0:
                    retracao_p = base.high[i-1] - base.close[i-1]
                    retracao_n = base.open[i-1] - base.low[i-1]
                if base.dif[i-1]<0:
                    retracao_p = base.high[i-1] - base.open[i-1]
                    retracao_n = base.close[i-1] - base.low[i-1]
                if base.dif[i-1]==0:
                    retracao_p = base.high[i-1] - base.close[i-1]
                    retracao_n = base.open[i-1] - base.low[i-1]
                dados = dados.append({'Hora': float(input_rnn[i-1][0][0]),
                                      'dif':float(input_rnn[i-1][0][1]),
                                      'retracao +':float(retracao_p),
                                      'retracao -':float(retracao_n),
                                      'RSI':float(input_rnn[i-1][0][4]),
                                      'M22M44':float(input_rnn[i-1][0][5]),
                                      'M22M66':float(input_rnn[i-1][0][6]),
                                      'M66M44':float(input_rnn[i-1][0][7]),
                                      'ADX':float(input_rnn[i-1][0][8]),
                                      'ATR':float(input_rnn[i-1][0][9]),
                                      'Momentum':float(input_rnn[i-1][0][10]), 
                                      'CCI':float(input_rnn[i-1][0][11]), 
                                      'Bears':float(input_rnn[i-1][0][12]), 
                                      'Bulls':float(input_rnn[i-1][0][13]), 
                                      'Stock1':float(input_rnn[i-1][0][14]),
                                      'Stock2':float(input_rnn[i-1][0][15]), 
                                      'Wilians':float(input_rnn[i-1][0][16]), 
                                      'Std':float(input_rnn[i-1][0][17]), 
                                      'MFI':float(input_rnn[i-1][0][18]),
                                      'target':int(target)}, ignore_index=True)
        return dados    