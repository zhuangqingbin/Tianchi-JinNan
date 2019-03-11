#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb



from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
#pd.set_option('display.max_columns', None) 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")




def stack_model(param_lgb,param_xgb,N,train_data,test_data,optimize_data):
    print('开始训练数据......')
    X_train = train_data.values
    y_train = target.values
    X_test = test_data.values
    optimize_values = optimize_data.values
    
    
    ### 训练ligthgbm
    print('训练ligthgbm......')
    param = param_lgb
    
    folds = KFold(n_splits=N, shuffle=False, random_state=2018)
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros(len(X_test))
    
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
    
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data],\
                        verbose_eval=200, early_stopping_rounds = 100)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        
        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    
    optimize_lgb = clf.predict(optimize_values, num_iteration=clf.best_iteration)[0]
    
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
    
    ### 训练xgboost
    ##### xgb
    print('训练xgboost......')
    xgb_params = param_xgb
    folds = KFold(n_splits=N, shuffle=True, random_state=2019)
    oof_xgb = np.zeros(len(X_train))
    predictions_xgb = np.zeros(len(X_test))
    
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx],)
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])
    
        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, \
                        verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
        
    optimize_xgb = clf.predict(xgb.DMatrix(optimize_values), ntree_limit=clf.best_ntree_limit)[0]
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))
    
    ### 融合训练
    # 将lgb和xgb的结果进行stacking
    train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
    
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()
    optimize_stack = np.vstack([np.array(optimize_lgb).reshape(1,), np.array(optimize_xgb).reshape(1,)]).transpose()
    
    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    optimze_predict = np.zeros(optimize_stack.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
        trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
        val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
        
        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)
        
        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
        optimze_predict += clf_3.predict(optimize_stack) / 10
    
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, target)))
    
    
    # 最优收率
    submit = pd.DataFrame({'id':'sample_0','target':optimze_predict}).\
    to_csv("submit_optimize.csv", index=False, header=None)
    
    # 复赛预测
    sub_df  = pd.read_csv('data/FuSai.csv', encoding = 'gb18030').rename({'样本id':'id'},axis=1)[['id','B14']]
    sub_df['target'] = predictions
    submit_df = map_by_regulation(sub_df,unique1_map,unique2_map,unique3_map)
    submit_df[['id','target']].to_csv("submit_FuSai.csv", index=False, header=None)
