# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:00:14 2017

@author: Lijie
"""
import logging
import logging.config

logging.basicConfig(level=logging.ERROR,#CRITICAL
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log',
                filemode='w')
    

#logging.config.fileConfig("logger.conf")

# 用来正常显示中文标签
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['SimHei']

import numpy as np
## 读数据
## 读数据
import pandas as pd 
data = pd.read_excel('E:\\上汽工作\\车享家库存\\yuqiaolu_top1.xlsx',header = 0)
##yuqiaolu_top1
##qishanlu_top1
##miaopulu_top1
##sanlinlu_top1
data.rename(columns={ 'num': '数量'}, inplace=True) 

########################  业务相关预处理  ############################
## 根据实际业务情形： 对数据进行数据汇总、分组等操作，处理为所需数据类型。
## 纪念路  黄帽子 不考虑
####################################################################
import pandas as pd 
def add_season(data):
    data['date'] = pd.to_datetime(data['year_week_id'])
    data['month'] = [int(i.month) for i in data['date']]
    season = []
    for i in data['month']:
        
        if i in [1,2,3]:
            season.append(1)
        elif i in [4,5,6]:
            season.append(2)
        elif i in [7,8,9]:
            season.append(3)
        else :
            season.append(4)
    data['season'] = season
    return data
 
data = add_season(data)     
        
        
def filter(data):
   
    ## 前两个周发货量的均值、最大值、最小值、增量作为输入
    d_mean = pd.rolling_mean(data['数量'],3).dropna()
    d_mean = pd.DataFrame(d_mean)
    d_mean.rename(columns = {'数量':'mean'}, inplace=True)
    
    ## 最大值
    d_max = pd.rolling_max(data['数量'],3).dropna()
    d_max = pd.DataFrame(d_max)
    d_max.rename(columns = {'数量':'max'}, inplace=True)
    
    ##最小值
    d_min = pd.rolling_min(data['数量'],3).dropna()
    d_min = pd.DataFrame(d_min)
    d_min.rename(columns = {'数量':'min'}, inplace=True)
    
    ## 方差
    d_std = pd.rolling_std(data['数量'],3).dropna()
    d_std = pd.DataFrame(d_std)
    d_std.rename(columns = {'数量':'std'}, inplace=True)
    
    ## 增量
    d_add12 = data['数量'] - data['数量'].shift(1)
    d_add12 = d_add12.dropna()
    d_add12 = pd.DataFrame(d_add12)
    d_add12.rename(columns = {'数量':'add12'}, inplace=True)
    
    
    d_add13 = data['数量'] - data['数量'].shift(2)
    d_add13 = d_add13.dropna()
    d_add13 = pd.DataFrame(d_add13)
    d_add13.rename(columns = {'数量':'add13'}, inplace=True)
    
    d_add23 = data['数量'].shift(1) - data['数量'].shift(2)
    d_add23 = d_add23.dropna()
    d_add23 = pd.DataFrame(d_add23)
    d_add23.rename(columns = {'数量':'add23'}, inplace=True)
    
    ## 数据合并及类型转换
    data_x = pd.merge(d_mean,d_max,left_index = True, right_index = True)
    data_x = pd.merge(data_x,d_min,left_index = True, right_index = True) 
    data_x = pd.merge(data_x,d_std,left_index = True, right_index = True) 
    data_x = pd.merge(data_x,d_add12,left_index = True, right_index = True) 
    data_x = pd.merge(data_x,d_add13,left_index = True, right_index = True) 
    data_x = pd.merge(data_x,d_add23,left_index = True, right_index = True) 
    data_x = pd.DataFrame(data_x).reset_index()
    del data_x['index']
    
    #data_x['season'] =data['season'].shift(2).dropna()
    
    ## 目标变量：下一时段需求量
    data_y = data.shift(-3)['数量'].dropna()
    data_y = pd.DataFrame(data_y).reset_index()
    del data_y['index']
    
    return data_x,data_y

data_x,data_y = filter(data)
#####################################################
## modeling 
## 
#####################################################
from sklearn import ensemble
from sklearn.metrics import mean_squared_error  
import xgboost as xgb 

## data_x 行数比 data_y多一行
## 数据 holdout
#data_x = pd.DataFrame(data_x['mean'])
#data_x = data_x[['mean','std','add12','add23']]
offset = data_x.shape[0]-5
X_train, y_train = data_x[:offset], data_y[:offset]
X_test, y_test = data_x[offset:-1],data_y[offset:]

## 模型融合
def combine(X_train, X_test, y_train,y_test):
    ## xgboost
    params = {'n_estimators': 100} #, 'learning_rate': 0.01}
    clf = xgb.XGBRegressor(**params)
    clf.fit(X_train, y_train) 
    print( '========Model 1 Fitted==========')
    pred1 = clf.predict(X_test)
  
    mse1 = mean_squared_error(y_test, pred1)  
    print ("MSE: %.4f" % mse1) 
    
    ## 随机森林
    rfc = ensemble.RandomForestRegressor(n_estimators=100, random_state=0)
    rfc.fit(X_train,y_train)
    print( '========Model 2 Fitted==========')
    pred2 = rfc.predict(X_test)
    mse2 = mean_squared_error(y_test, pred2)  
    print ("MSE: %.4f" % mse2) 

    ## 融合预测及整体误差
    pred = ( mse2 * pred1 + mse1 * pred2)/( mse1+ mse2)
   # mse = 1-np.abs(pred['数量']- y_test['数量'])/(y_test['数量']+0.1)
    print( '========Predict Finished======')
    print('ensemble_error',  mean_squared_error(y_test, pred)  )
    
    ##　单样本误差
    pred = pd.DataFrame(pred).reset_index().rename(columns={ 0: '预测值'})
    pred['预测值'] = pred['预测值'].astype(int)
    del pred['index']
    y_test = pd.DataFrame(y_test).reset_index().rename(columns={ '数量': '真实值'})
    del y_test['index']
    pred['真实值'] = y_test['真实值']
    
    
    pred['误差'] = 1-np.abs(pred['真实值']- pred['预测值'])/(pred['真实值']+0.01)
    return pred


print('combine: ')
pred = combine(X_train, X_test, y_train,y_test)  

