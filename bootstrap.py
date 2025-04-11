# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:16:18 2018

@author: LP
"""

# 导入库
import os
from sklearn.utils import resample
from tqdm import tqdm
import numpy as np  # numpy库
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor # 决策树回归
from sklearn.ensemble import RandomForestRegressor # 随机森林回归
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score,mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
# 数据准备
#k1 = np.loadtxt("C:\\Users\\lenovo\\Desktop\\5ego1.5.txt",dtype=float)
# k1 = np.loadtxt("C:\\Users\\Administrator\\Desktop\\刘祖罡\\数据整理4\\Ptslow\\Ptslow1.csv",dtype=float,delimiter=',')

k1 = pd.read_excel('训练集与测试集.xlsx',sheet_name='Sheet1',header=0,index_col=0)  # 数据名称
k3 = pd.read_excel('训练集与测试集.xlsx',sheet_name='Sheet2',header=0,index_col=0)  # 数据名称

# all_data = k1.replace(' ', np.nan)
# k1 = all_data.dropna()
#
#
# k1 = k1.astype("float")
# k1 = np.array(k1)
#
#
# # minmax = MinMaxScaler()
# # X = minmax.fit_transform(x)
#
# t1 = k1.shape[0]
#
# k3 = k1  # 数据名称
# k3 = np.array(k3)

# 训练回归模型
##### 1、线性回归
param_grid_lr = [{}]
model_lr = LinearRegression()
##### 2、 支持向量机回归
k_range = np.arange(100,1001,100)
k2_range = np.arange(0.01,0.11,0.01)
param_grid_svr = dict(C = k_range,gamma = k2_range)
model_svr = SVR(kernel='rbf')
##### 3、 自适应集成回归
param_grid_abr = [{'learning_rate': [0.1,0.5,1,10], 'n_estimators': [10,100]}]
model_abr = AdaBoostRegressor()  # 建立自适应增强回归模型对象
#### 4、 近邻回归
param_grid_nei = [{'n_neighbors': [1,2,3,4,5,6,7,8,9],'leaf_size': [1,5,10,20,30,40,50,60,70,80,90,100]}]
model_nei = neighbors.KNeighborsRegressor(weights='distance')  # k近邻回归
##### 5、 决策树回归
param_grid_dtr = [{'min_samples_split': [2,3,4,5,6,7,8,9],'min_samples_leaf': [1,5,10,20,30,40,50,60,70,80,90,100]}]
model_dtr = DecisionTreeRegressor(max_depth=7) # 决策树回归
##### 6、 随机森林回归
param_grid_rfr = [{'n_estimators': [10,100],'min_samples_split': [2,5,10,20],'min_samples_leaf': [1,5,10]}]
model_rfr = RandomForestRegressor()
# #### 7、 梯度提升回归
# param_grid_gbr = [{'learning_rate': [0.1,0.5,1,10], 'n_estimators': [10,100],
#                    'max_depth': [1,3,5,10,20], 'min_samples_split': [2,5,10,20],
#                    'min_samples_leaf': [1,5,10]}]
# model_gbr = GradientBoostingRegressor(random_state = 0)
##### 8、 贝叶斯岭回归
param_grid_br = [{'n_iter': [10,30,50,100], 'tol': [1,5,10], 'lambda_2': [1e-02,1e-01,1]}] #  'alpha_1': [1e-04,1e-03,1e-02,1e-01,1], 'lambda_1': [1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1], 'alpha_2': [1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1],
model_br = BayesianRidge()

###### 10、神经网络回归
#param_grid_mlp = [{'alpha': [0.0001,0.001,0.01,0.1], 'learning_rate_init': [0.001,0.01,0.1],
#                  'max_iter': [1000], 'tol': [0.0001,0.001,0.01,0.1],
#                  'power_t': [0.1,0.5,1]}]
#model_mlp = MLPRegressor()

model_names = ['RFR']  # 不同模型的名称列表'LR','ABR','NEI','DTR','RF', ,'GBR','BR','La'   'MLP','SVR'
model_dic = [model_rfr]  # 不同回归模型对象的集合model_lr, model_abr, model_nei, model_dtr, model_rfr,, model_gbr, model_br, model_La  model_MLP,
param_grid = [param_grid_rfr]  #param_grid_lr, param_grid_abr, param_grid_nei, param_grid_dtr, param_grid_rfr,, param_grid_gbr, param_grid_br, param_grid_la   param_grid_mlp,

#model_names = ['MLP']  # 不同模型的名称列表'LR','ABR','NEI','DTR','RF', ,'GBR','BR','La'   'MLP','SVR'
#model_dic = [model_mlp]  # 不同回归模型对象的集合model_lr, model_abr, model_nei, model_dtr, model_rfr,, model_gbr, model_br, model_La  model_MLP,
#param_grid = [param_grid_mlp]  #param_grid_lr, param_grid_abr, param_grid_nei, param_grid_dtr, param_grid_rfr,, param_grid_gbr, param_grid_br, param_grid_la   param_grid_mlp,

y_plot = np.arange(2)
cv_score_list =[]
tmp_list = []
#temp = []
z = 0
for model,param in zip(model_dic, param_grid):  # 读出每个回归模型对象
    exp__score,r2__s,mse__score,mae__score,ss= np.arange(1),np.arange(1),np.arange(1),np.arange(1),np.arange(1)
    temp = []
    for f in tqdm(range(100), unit='次', desc='time'):

        X_train = k1.iloc[:,:-1]
        y_train = k1.iloc[:,-1]
        X_test = k3.iloc[:,:-1]
        y_test = k3.iloc[:,-1]
        grid = GridSearchCV(model, param, cv=10,scoring='neg_mean_squared_error')  #neg_mean_squared_error
        grid.fit(X_train, y_train)
        grid_est = grid.best_estimator_
        grid_par = grid.best_params_
        model = grid_est.fit(X_train, y_train)
        scores_train = cross_val_score(model, X_train, y_train,cv=10,scoring='neg_mean_squared_error') #训练集交叉验证
        scores_train_mean = np.mean(scores_train) # 训练集1次10折交叉验证后取平均值
        ss = np.row_stack((ss,scores_train_mean))
        # y_pred = model.predict(X_train)
        y_pred = model.predict(X_test)
        # yplot = np.column_stack((y_train,y_pred))
        yplot = np.column_stack((y_test, y_pred))
        y_plot = np.row_stack((y_plot,yplot))
        # exp_score = explained_variance_score(y_train,y_pred)
        exp_score = explained_variance_score(y_test, y_pred)
        exp__score = np.row_stack((exp__score,exp_score))
        # r2_s = r2_score(y_train,y_pred)
        r2_s = r2_score(y_test, y_pred)
        r2__s = np.row_stack((r2__s,r2_s))
        # mse_score = mean_squared_error(y_train,y_pred)
        mse_score = mean_squared_error(y_test, y_pred)
        mse__score = np.row_stack((mse__score,mse_score))
        # mae_score = mean_absolute_error(y_test, y_pred)
        mae_score = mean_absolute_error(y_test, y_pred)
        mae__score = np.row_stack((mae__score, mae_score))
    sss = np.mean(ss[1:,:])
    cv_score_list.append(scores_train)
    Exp = np.mean(exp__score[1:,:])
    Expstd = np.std(exp__score[1:,:])
    R2 = np.mean(r2__s[1:,:])
    R2std = np.std(r2__s[1:,:])
    MSE = np.mean(mse__score[1:,:])
    MSEstd = np.std(mse__score[1:,:])
    MAE = np.mean(mae__score[1:,:])
    MAEstd = np.std(mae__score[1:,:])
    temp.append(Exp)
    temp.append(Expstd)
    temp.append(R2)
    temp.append(R2std)
    temp.append(MSE)
    temp.append(MSEstd)
    temp.append(MAE)
    temp.append(MAEstd)
    temp = np.array(temp)
    tmp_list.append(temp)
    del temp
    del exp_score
    del exp__score
    del r2_s
    del r2__s
    del mse_score
    del mse__score
    z += 1
    print('-- Mission %d Complete --' %z )

# 模型效果指标评估
df1 = pd.DataFrame(cv_score_list,index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(tmp_list, index=model_names, columns=['EV','EVstd','R2','R2std','MSE','MSEstd','MAE','MAEstd'])  # 建立回归指标的数据框
#print (70 * '-')  # 打印分隔线
#print ('cross validation result:')  # 打印输出标题
#print (df1)  # 打印输出交叉检验的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')  # 打印分隔线
df2.to_excel(r'预测结果.xlsx' , float_format = '%.12f')










