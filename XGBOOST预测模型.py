# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:26:51 2017

@author: Lijie
"""

# 用来正常显示中文标签
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['SimHei']

import numpy as np
## 读数据
import pickle 
f1 = open('E:\\上汽工作\\上汽国际\\data\\实际发货明细.pkl','rb')
data = pickle.load(f1)
f1.close()

########################  常规预处理  ############################
## 数据去重复
## 缺失值处理
################################################################
def preprocessFirst(data):
    ## 1 数据去重复
    data = data.drop_duplicates()  
    ## 2 缺失值处理
    cnt = data['数量'].isnull().value_counts()  ## 缺失量
    data.dropna()
    return data
#data = preprocessFirst(data)



########################  业务相关预处理  ############################
## 根据实际业务情形： 对数据进行数据汇总、分组等操作，处理为所需数据类型。
## 
####################################################################
import pandas as pd 
def filter(data, materialcode = 200300001):
    #data = data.drop_duplicates()  ## 去掉重复值
    data = data[data['物料'] == materialcode ].copy()
    weekNum = data[['工厂','实际发货过账所在周','数量']].groupby(['实际发货过账所在周','工厂']).sum()
    deliveryTime = data[['工厂','实际发货过账所在周','实际发货过账日期']].groupby(['实际发货过账所在周','工厂']).max()
    modelData = weekNum.join(deliveryTime, how = 'left')  ## 物料数据： 建模数据
    modelData = modelData.reset_index()
    modelData = modelData[['实际发货过账所在周','数量']]
    
    ## 前两个周发货量的均值作为输入
    data_x = pd.rolling_mean(modelData,2)['数量'].dropna()
    #data_x = pd.rolling(center=False,window=4).mean()['数量'].dropna()
    data_y = modelData.shift(2)['数量'].dropna()
    
    return data_x,data_y

data_x,data_y = filter(data)
data_x = pd.DataFrame(data_x)
data_y = pd.DataFrame(data_y)


##########################################################
##data就是一个普通的dataframe格式，其中'y'就是因变量，然后可以直接fit拟合函数。
##最后，输出mean_squared_error平方误差，衡量模型预测好坏。
##
############################################################


#XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, **kwargs)  
#from sklearn.metrics import confusion_matrix, mean_squared_error  
import xgboost as xgb  

gbm = xgb.XGBRegressor().fit(data_x[:-10], data_y[:-9])  
predictions = gbm.predict(data_x[:-1]) 

predictions = pd.DataFrame(predictions)
predictions.rename(columns={ 0: '数量'}, inplace=True)  



#print('---- RMSE: 均方根误差 ----')
#print(np.sqrt(mean_squared_error(actuals, predictions)))  

actuals = data_y.reset_index() 
predictions = predictions.reset_index()


#### 测试集上的表现
def validation(predictions,actuals, d):
    test = pd.DataFrame()
    
    test['真实值'] = actuals.tail(d)['数量']
    test['预测值'] = predictions.tail(d)['数量']
    test['精度'] = 1-np.abs(predictions['数量']- actuals['数量'])/actuals['数量']

    return test
    
    

test = validation(predictions,actuals, 8)
print('--------------------------------------------')
print('平均误差', test['精度'].mean())
        




import matplotlib.pyplot as plt 
## 预测与历史数据同框展示
plt.figure(facecolor='white',figsize=(12, 8))
plt.ylim(0,500000)
predictions = pd.DataFrame(predictions)


plt.plot(predictions['数量'],color='green', label='Predict')
#predictions.plot(color='green', label='Predict')

plt.plot(actuals['数量'],color='orange', label='Original')
#real_data.plot(color='orange', label='Original')
plt.legend(loc='best')
#tt = predictions.reset_index()['数量']-real_data.reset_index()['数量']
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions['数量']- actuals['数量'])**2)/actuals.size))
plt.show()

