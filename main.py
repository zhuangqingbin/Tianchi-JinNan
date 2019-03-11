#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,sys,gc
sys.path.append('codes')
sys.path.append('data')
import numpy as np 
import pandas as pd 


from DataRead import read_data
from DataModify import modify_train
from DataProcess import process_data
from Model_stack import stack_model
from configuration import param_lgb,param_xgb


# 读取数据
train_data,test_data,optime_data = read_data()

# 修正数据
train_data = modify_train(train_data)

# 特征工程
train_data,test_data,optime_data = process_data(train_data,test_data,optime_data)



### 训练
stack_model(param_lgb=param_lgb,param_xgb=param_xgb,N=5,train_data,test_data,optimize_data)
