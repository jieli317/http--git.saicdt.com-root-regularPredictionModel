# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:31:20 2017

@author: Lijie
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt
 
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf        
import pickle

#有中文出现的情况，需要u'内容'
# -*- coding: utf-8 -*-
import matplotlib 
chfont = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc') #'/usr/share/fonts/cjkunifonts-ukai/ukai.ttc')


## 读数据
#data = pd.read_excel('E:\\上汽工作\\上汽国际\\data\\实际发货明细.xlsx',header = 0)
#
#f = open('E:\\上汽工作\\上汽国际\\data\\实际发货明细.pkl','wb')
#pickle.dump(data,f)
#f.close()


#f1 = open('E:\\上汽工作\\上汽国际\\data\\实际发货明细.pkl','rb')
#data = pickle.load(f1)
#f1.close()
###########################################################################
## 1 数据预处理： 转化为时间序列；
## 2 平稳根检验，自相关系数，偏自相关系数，残差正态性检验；
## 3 平滑、差分处理为平稳序列
## 4 ARMA算法训练
## 5 计算最优ARMA模型、 pdq 参数自动选优;
## 6 模型预测 
## 7 预测值差分、平滑等还原
## 8 滚动预测
#########################################################################
class arima_model:

    def __init__(self, data, maxLag=3, materialcode = 200300001 ,smoothCycle = 52 , d = [52,1]):
        self.data = data
        self.ts = None
        self.data_ts = None   ## 时间序列
        self.resid_ts = None  ## 残差， ma回归
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxsize
        self.materialcode = materialcode
        self.smoothCycle = smoothCycle
        self.d = d  ## 差分
    
    ## 数据预处理：按周合并，并提取各周对应时间属性。
    def filter(self):
        weekNum = self.data[['物料','工厂','实际发货过账所在周','数量']].groupby(['物料','实际发货过账所在周','工厂']).sum()
        deliveryTime = self.data[['物料','工厂','实际发货过账所在周','实际发货过账日期']].groupby(['物料','实际发货过账所在周','工厂']).max()
        modelData = weekNum.join(deliveryTime, how = 'left')  ## 物料数据： 建模数据
        modelData = modelData.reset_index()
        ## 指定物料数据
        wl = modelData[modelData['物料']== self.materialcode]
        
        wl.index = pd.to_datetime(wl['实际发货过账日期'])
        self.ts = wl['数量'][145:-1]
        initdata = wl['数量'][145:-1].plot()
        
        self.data_ts = wl['数量'][145:-1]
        print('真实值：', wl['数量'][-6:])
        return self.ts,initdata
    
    ## 差分操作
    def diff_ts(self):
        global shift_ts_list
        #  动态预测第二日的值时所需要的差分序列
        global last_data_shift_list
        shift_ts_list = []
        last_data_shift_list = []
        tmp_ts = self.ts
        for i in self.d:
            if i==0:
                last_data_shift_list.append(0)
                shift_ts_list.append(tmp_ts)
            else:
                last_data_shift_list.append(tmp_ts[-i])
                print (last_data_shift_list)
                shift_ts = tmp_ts.shift(i)
                shift_ts_list.append(shift_ts)
            tmp_ts = tmp_ts - shift_ts
        tmp_ts.dropna(inplace=True)
        self.ts = tmp_ts
        self.data_ts = tmp_ts
        return tmp_ts
    
    # 还原操作
    def predict_diff_recover(self, predict_value):
        if isinstance(predict_value, float):
            tmp_data = predict_value
            for i in range(len(self.d)):
                tmp_data = tmp_data + last_data_shift_list[-i-1]
        elif isinstance(predict_value, np.ndarray):
            tmp_data = predict_value[0]
            for i in range(len(self.d)):
                tmp_data = tmp_data + last_data_shift_list[-i-1]
        else:
            tmp_data = predict_value
            for i in range(len(self.d)):
                try:
                    tmp_data = tmp_data.add(shift_ts_list[-i-1])
                except:
                    raise ValueError('What you input is not pd.Series type!')
            tmp_data.dropna(inplace=True)
        return tmp_data

    # 计算最优ARIMA模型，将相关结果赋给相应属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()
                    

#    # 参数确定模型
#    def certain_model(self, p, q):
#            model = ARMA(self.data_ts, order=(p, q))
#            try:
#                self.properModel = model.fit( disp=-1, method='css')
#                self.p = p
#                self.q = q
#                self.bic = self.properModel.bic
#                self.predict_ts = self.properModel.predict()
#                self.resid_ts = deepcopy(self.properModel.resid)
#            except:
#                print ('You can not fit the model with this parameter p,q, ' \
#                      'please use the get_proper_model method to get the best model')

    # 预测第二日的值
    def forecast_next_day_value(self, type='day'):
        # 我修改了statsmodels包中arima_model的源代码，添加了constant属性，需要先运行forecast方法，为constant赋值
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params

        # print self.properModel.params
        if self.p == 0:   # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)
        #print('111',self.properModel.params[0])
        predict_value = np.dot(para[1:], values) + self.properModel.params[0] #.constant[0] #_intercept #
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    # 动态添加数据函数，针对索引是月份和日分别进行处理
    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat

    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)
        
    
   
f1 = open('E:\\上汽工作\\上汽国际\\data\\实际发货明细.pkl','rb')
data = pickle.load(f1)
f1.close()
##200300101,200300001,200300083,200300084,200300102
model = arima_model(data, maxLag=3, materialcode =200300001 ,smoothCycle = 52 , d = [52,0])
ts,initdata = model.filter()
model.diff_ts()
model.get_proper_model()
##tt = model.predict_ts(model.ts.values.tolist())

predict_value =model.forecast_next_day_value()
predict =  model.predict_diff_recover(predict_value)
#model.resid_ts.plot(color='red', label = 'ts')
#
#model.ts.plot(color='black', label = 'ts')
## 预测值复原
recover_pret = model.predict_diff_recover(model.predict_ts)
recover_pret.plot(color = 'black', label = 'recover')
initdata.plot(color ='red',label = 'ab')
#print('预测值',predict)

#foreCast =model.properModel.forecast(4)




#class preprocess:
#    def __init__(self,data, materialcode = 200300083 ,smoothCycle = 52 , d = 1):
#        self.data = data 
#        self.materialcode = materialcode ## 物料编码
#        self.smoothCycle = smoothCycle   ## 平滑周期
#        self.d = d  ## 差分阶数
#        
#    ## 数据预处理：按周合并，并提取各周对应时间属性。
#    def filter(self):
#        weekNum = self.data[['物料','工厂','实际发货过账所在周','数量']].groupby(['物料','实际发货过账所在周','工厂']).sum()
#        deliveryTime = self.data[['物料','工厂','实际发货过账所在周','实际发货过账日期']].groupby(['物料','实际发货过账所在周','工厂']).max()
#        modelData = weekNum.join(deliveryTime, how = 'left')  ## 物料数据： 建模数据
#        modelData = modelData.reset_index()
#        ## 指定物料数据
#        wl = modelData[modelData['物料']== self.materialcode]
#        
#        wl.index = pd.to_datetime(wl['实际发货过账日期'])
#        ts = wl['数量']
#        return ts
#    # 移动平均图
#    def draw_trend(self,timeSeries, size):
#        f = plt.figure(facecolor='white')
#        # 对size个数据进行移动平均
#        rol_mean = timeSeries.rolling(window=size).mean()
#        # 对size个数据进行加权移动平均
#        rol_weighted_mean = pd.ewma(timeSeries, span=size)
#    
#        timeSeries.plot(color='blue', label='Original')
#        rol_mean.plot(color='red', label='Rolling Mean')
#        rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
#        plt.legend(loc='best')
#        plt.title('Rolling Mean')
#        plt.show()
#    
#    def draw_ts(self,timeSeries):
#        f = plt.figure(facecolor='white')
#        timeSeries.plot(color='blue')
#        plt.show()
#    
#    '''
#    　　Unit Root Test
#       The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
#       root, with the alternative that there is no unit root. That is to say the
#       bigger the p-value the more reason we assert that there is a unit root
#       --Augmented Dickey–Fuller test 扩展迪基-福勒检验: 用于测试平稳性(单位根检验)
#       Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Valuesfor difference confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary. Refer this article for details.
#    '''
#    def testStationarity(self,ts):
#        dftest = adfuller(ts)
#        # 对上述函数求得的值进行语义描述
#        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#        for key,value in dftest[4].items():
#            dfoutput['Critical Value (%s)'%key] = value
#        return dfoutput
#    
#    # 自相关和偏相关图，默认阶数为31阶
#    def draw_acf_pacf(self,ts, lags=31):
#        f = plt.figure(facecolor='white')
#        ax1 = f.add_subplot(211)
#        plot_acf(ts, lags=lags, ax=ax1)
#        ax2 = f.add_subplot(212)
#        plot_pacf(ts, lags=lags, ax=ax2)
#        plt.show()
#    
#    def main(self):
#        ts = self.filter()
#        rol_mean = ts.rolling(window = self.smoothCycle).mean()
#        rol_mean.dropna(inplace=True)
#        rol_mean.plot()
#                
#        ts_diff_1 = rol_mean.diff(1)
#        ts_diff_1.dropna(inplace=True)
#        ts_diff_2 = rol_mean.diff(1)
#        ts_diff_2.dropna(inplace=True)
#        
#        ##df = self.testStationarity(ts_diff_2)
#        ##self.draw_acf_pacf(ts_diff_2)
#        
#        return ts_diff_2
    
## 差分操作
#def diff_ts(ts, d):
#    global shift_ts_list
#    #  动态预测第二日的值时所需要的差分序列
#    global last_data_shift_list
#    shift_ts_list = []
#    last_data_shift_list = []
#    tmp_ts = ts
#    for i in d:
#        last_data_shift_list.append(tmp_ts[-i])
#        print (last_data_shift_list)
#        shift_ts = tmp_ts.shift(i)
#        shift_ts_list.append(shift_ts)
#        tmp_ts = tmp_ts - shift_ts
#    tmp_ts.dropna(inplace=True)
#    return tmp_ts
#
##tmp_ts = diff_ts(ts,[4,1])
## 还原操作
#def predict_diff_recover(predict_value, d):
#    if isinstance(predict_value, float):
#        tmp_data = predict_value
#        for i in range(len(d)):
#            tmp_data = tmp_data + last_data_shift_list[-i-1]
#    elif isinstance(predict_value, np.ndarray):
#        tmp_data = predict_value[0]
#        for i in range(len(d)):
#            tmp_data = tmp_data + last_data_shift_list[-i-1]
#    else:
#        tmp_data = predict_value
#        for i in range(len(d)):
#            try:
#                tmp_data = tmp_data.add(shift_ts_list[-i-1])
#            except:
#                raise ValueError('What you input is not pd.Series type!')
#        tmp_data.dropna(inplace=True)
#    return tmp_data
        

#df = prep.testStationarity(ts)