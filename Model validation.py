#!/usr/bin/env python
# coding: utf-8

# # out-of-sample validation

# In[ ]:


#导入库
import pandas as pd
import numpy as np
import os
import time
import psutil
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold   #k折交叉验证  
from sklearn.model_selection import train_test_split #训练集和测试集拆分
from sklearn.preprocessing import MinMaxScaler  #最小最大归一化
from sklearn.metrics import mean_squared_error #mse
from sklearn.metrics import mean_absolute_error #mae
from sklearn.metrics import r2_score #r2
import math 
import pickle
from itertools import product 


# In[ ]:


data=pd.read_excel("/tune/out-of-sample/out-of-sample data.xlsx",header=0,index_col=None)
data


# In[ ]:


#only use without space-time
data.drop(["distance","doy","hour"],axis=1,inplace=True)
data


# In[ ]:


features = data.columns[:-1]
target = data.columns[-1]


# In[ ]:


kf = KFold(n_splits=10, random_state=17, shuffle=True)


# In[ ]:


n_estimators_range = [200,400,600,800,1000]
learning_rate_range = [0.01,0.05,0.1,0.2,0.3]
num_leaves_range = [15,31,63,127]
max_depth_range=[-1,2,4,6,8,10]
min_child_samples_range = [5,10,15,20,25,30]
reg_lambda_range = [0,5,10,15,20,30]
reg_alpha_range = [0,5,10,15,20,30]
param_grid = product(n_estimators_range, learning_rate_range, num_leaves_range,max_depth_range, min_child_samples_range, reg_lambda_range, reg_alpha_range)
best_test_rmse = float('inf')
best_test_r2 = -1
best_test_mae = float('inf')
best_train_rmse = float('inf')
best_train_r2 = -1
best_train_mae = float('inf')
best_params = None
process = psutil.Process(os.getpid())


# In[ ]:


#建立空列表保存性能值
results = {
    'n_estimators': [],
    'learning_rate': [],
    'num_leaves': [],
    'max_depth':[],
    'min_child_samples': [],
    'reg_lambda': [],
    'reg_alpha':[],
    'train_rmse_mean': [],
    'test_rmse_mean': [],
    'train_r2_mean': [],
    'test_r2_mean': [],
    'train_mae_mean': [],
    'test_mae_mean': [],
    'runtime':[],
    'mem_usage':[],
}

for params in param_grid:
    #建立空列表，存储每一折上的性能
    train_rmses = []
    train_r2s = []
    train_maes = []
    test_rmses= []
    test_r2s= []
    test_maes= []
    
    n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda, reg_alpha = params
    results['n_estimators'].append(n_estimators)
    results['learning_rate'].append(learning_rate)
    results['num_leaves'].append(num_leaves)
    results['max_depth'].append(max_depth)
    results['min_child_samples'].append(min_child_samples)
    results['reg_lambda'].append(reg_lambda)
    results['reg_alpha'].append(reg_alpha)
    print(f"now：n_estimators：{n_estimators},learning_rate：{learning_rate},num_leaves:{num_leaves},max_depth:{max_depth},min_child_sample：{min_child_samples},reg_lambda：{reg_lambda},reg_alpha:{reg_alpha}")
    
    start_time = time.time
    start_mem = process.memory_info().rss 
    for train_index, test_index in kf.split(data):
        train_X = data.iloc[train_index][features]
        train_y = data.iloc[train_index][target]
        test_X = data.iloc[test_index][features]
        test_y = data.iloc[test_index][target]       

        model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, 
                              max_depth=max_depth, min_child_samples=min_child_samples, reg_lambda=reg_lambda, 
                              reg_alpha=reg_alpha)
        model.fit(train_X, train_y)
        train_pred= model.predict(train_X)
        test_pred = model.predict(test_X)
        
        train_rmses.append(np.sqrt(mean_squared_error(train_y, train_pred)))
        train_r2s.append(r2_score(train_y,train_pred))
        train_maes.append(mean_absolute_error(train_y,train_pred))
        test_rmses.append(np.sqrt(mean_squared_error(test_y, test_pred)))
        test_r2s.append(r2_score(test_y,test_pred))
        test_maes.append(mean_absolute_error(test_y,test_pred))
    
    end_time = time.time() 
    end_mem = process.memory_info().rss
    runtime = end_time - start_time
    mem_usage = (end_mem - start_mem)/ (1024**2)

    train_rmse_mean = np.mean(train_rmses)
    train_r2_mean=np.mean(train_r2s)
    train_mae_mean=np.mean(train_maes)
    test_rmse_mean=np.mean(test_rmses)
    test_r2_mean=np.mean(test_r2s)
    test_mae_mean=np.mean(test_maes)
    results['train_rmse_mean'].append(train_rmse_mean)
    results['train_r2_mean'].append(train_r2_mean)
    results['train_mae_mean'].append(train_mae_mean)
    results['test_rmse_mean'].append(test_rmse_mean)
    results['test_r2_mean'].append(test_r2_mean)
    results['test_mae_mean'].append(test_mae_mean)
    results['runtime'].append(runtime)
    results['mem_usage'].append(mem_usage)
    print(f"now train_rmse:{train_rmse_mean.item():.3f},train_r2:{train_r2_mean.item():.3f},train_mae:{train_mae_mean.item():.3f}")
    print(f"now test_rmse:{test_rmse_mean.item():.3f},test_r2:{test_r2_mean.item():.3f},test_mae:{test_mae_mean.item():.3f}")

df=pd.DataFrame(results)
name="/tune/out-of-sample/lgb performance.xlsx"
df.to_excel(name,header=True,index=True)


# In[ ]:


# train the final model
train_rmses = []
train_r2s = []
train_maes = []
test_rmses= []
test_r2s= []
test_maes= []
for train_index, test_index in kf.split(data):
    train_X = data.iloc[train_index][features]   #X是输入变量集
    train_y = data.iloc[train_index][target]   #voc一列
    test_X = data.iloc[test_index][features]
    test_y = data.iloc[test_index][target]

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.2, num_leaves=63, 
                            max_depth=10,min_child_samples=5)
    model.fit(train_X, train_y)
    train_pred= model.predict(train_X)
        
    train_rmses.append(np.sqrt(mean_squared_error(train_y, train_pred)))
    train_r2s.append(r2_score(train_y,train_pred))
    train_maes.append(mean_absolute_error(train_y,train_pred))
    test_rmses.append(np.sqrt(mean_squared_error(test_y, test_pred)))
    test_r2s.append(r2_score(test_y,test_pred))
    test_maes.append(mean_absolute_error(test_y,test_pred))
        
train_rmse_mean = np.mean(train_rmses)
train_r2_mean=np.mean(train_r2s)
train_mae_mean=np.mean(train_maes)
test_rmse_mean=np.mean(test_rmses)
test_r2_mean=np.mean(test_r2s)
test_mae_mean=np.mean(test_maes)


# In[ ]:


pickle.dump(model,open("/tune/out-of-sample/out-of-sample stlgb.dat","wb"))


# # out-of-station validation

# In[ ]:


data=pd.read_excel("/tune/out-of-station/out-of-station data.xlsx",header=0,index_col=None)
data


# In[ ]:


data["grid"].unique()


# In[ ]:


zhandian=[ 1155,  2020,  2376,  2594,  2597,  2788,  2820,  3043,  4005,
        5704,  5814,  5815,  6000,  6035,  7980,  8209,  8306,  8309,
        8318,  8424,  8426,  8641,  8645,  8965,  8970,  8972,  9068,
        9077,  9290,  9502,  9613,  9615,  9723,  9725,  9728,  9833,
        9834,  9835,  9838,  9943, 10057, 10058]


# In[ ]:


num1=random.sample(zhandian,12)
num2=[item for item in zhandian if item not in num1]
df_test=pd.DataFrame()
df_train=pd.DataFrame()
for m in num1:
    zan1=data[data['grid'].isin([m])]
    df_test=pd.concat([df_test,zan1],axis=0)
    
for n in num2:
    zan2=data[data['grid'].isin([n])]
    df_train=pd.concat([df_train,zan2],axis=0)
    
df_train=df_train.drop("grid",axis=1)
df_test=df_test.drop("grid",axis=1)
train_X=df_train.iloc[:,:-1]
train_y=df_train.iloc[:,-1]
test_X=df_test.iloc[:,:-1]
test_y=df_test.iloc[:,-1]


# In[ ]:


# 设置参数范围
n_estimators_range = [200,400,600,800,1000]
learning_rate_range = [0.01,0.05,0.1,0.2,0.3]
num_leaves_range = [15,31,63,127]
max_depth_range=[-1,2,4,6,8,10]
min_child_samples_range = [5,10,15,20,25,30]
reg_lambda_range = [0,5,10,15,20,30]
reg_alpha_range = [0,5,10,15,20,30]
param_grid = product(n_estimators_range, learning_rate_range, num_leaves_range,
                     max_depth_range, 
                     min_child_samples_range, reg_lambda_range, reg_alpha_range)

# 初始化最优评估指标
best_test_rmse = float('inf')
best_test_r2 = -1
best_test_mae = float('inf')
best_train_rmse = float('inf')
best_train_r2 = -1
best_train_mae = float('inf')
best_params = None
process = psutil.Process(os.getpid())


# In[ ]:


#建立空列表保存性能值
results = {
    'n_estimators': [],
    'learning_rate': [],
    'num_leaves': [],
    'max_depth':[],
    'min_child_samples': [],
    'reg_lambda': [],
    'reg_alpha':[],
    'train_rmse': [],
    'test_rmse': [],
    'train_r2': [],
    'test_r2': [],
    'train_mae': [],
    'test_mae': [],
    'runtime':[],
    'mem_usage':[],
}


# In[ ]:


for params in param_grid:
    n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda, reg_alpha = params
    results['n_estimators'].append(n_estimators)
    results['learning_rate'].append(learning_rate)
    results['num_leaves'].append(num_leaves)
    results['max_depth'].append(max_depth)
    results['min_child_samples'].append(min_child_samples)
    results['reg_lambda'].append(reg_lambda)
    results['reg_alpha'].append(reg_alpha)
    print(f"now：n_estimators：{n_estimators},learning_rate：{learning_rate},num_leaves:{num_leaves},max_depth:{max_depth},min_child_sample：{min_child_samples},reg_lambda：{reg_lambda},reg_alpha:{reg_alpha}")
：{reg_lambda},reg_alpha:{reg_alpha}")
    
    start_time = time.time()  
    start_mem = process.memory_info().rss  
    model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, 
                          max_depth=max_depth,
                          min_child_samples=min_child_samples, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    model.fit(train_X, train_y)
    train_pred= model.predict(train_X)
    test_pred = model.predict(test_X)
    
    end_time = time.time() 
    end_mem = process.memory_info().rss  
    runtime = end_time - start_time 
    mem_usage = (end_mem - start_mem)/ (1024**2)
    
    train_rmse=np.sqrt(mean_squared_error(train_y, train_pred))
    train_r2=r2_score(train_y,train_pred)
    train_mae=mean_absolute_error(train_y,train_pred)
    test_rmse=np.sqrt(mean_squared_error(test_y, test_pred))
    test_r2=r2_score(test_y,test_pred)
    test_mae=mean_absolute_error(test_y,test_pred)

    results['train_rmse'].append(train_rmse)
    results['train_r2'].append(train_r2)
    results['train_mae'].append(train_mae)
    results['test_rmse'].append(test_rmse)
    results['test_r2'].append(test_r2)
    results['test_mae'].append(test_mae)
    results['runtime'].append(runtime)
    results['mem_usage'].append(mem_usage)
    print(f"now train_rmse:{train_rmse.item():.3f},train_r2:{train_r2.item():.3f},train_mae:{train_mae.item():.3f}")
    print(f"now test_rmse:{test_rmse.item():.3f},test_r2:{test_r2.item():.3f},test_mae:{test_mae.item():.3f}")

df=pd.DataFrame(results)
name="/tune/out-of-station/lgb performance.xlsx"
df.to_excel(name,header=True,index=True)


# In[ ]:


pickle.dump(model,open("/tune/out-of-station/out-of-station stlgb.dat","wb"))

