#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:48:31 2019

@author: jimmy
"""

param_lgb = {
    'num_leaves': 120,
     'min_data_in_leaf': 30, 
     'objective':'regression',
     'max_depth': -1,
     'learning_rate': 0.005,
     "min_child_samples": 35,
     "boosting": "gbdt",
     "feature_fraction": 0.9,
     "bagging_freq": 1,
     "bagging_fraction": 0.9 ,
     "bagging_seed": 11,
     "metric": 'rmse',
     "lambda_l1": 0.1,
     "verbosity": -1
     }


param_xgb = {'eta': 0.005, 
             'max_depth': 12, 
             'subsample': 0.8, 
             'colsample_bytree': 0.8, 
             'objective': 'reg:linear', 
             'eval_metric': 'rmse', 
             'silent': True, 
             'nthread': 4
             }
