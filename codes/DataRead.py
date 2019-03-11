#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:34:02 2019

@author: jimmy
"""

def read_data():
    train = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding = 'gb18030').\
rename({'样本id':'id','收率':'target'},axis=1).pipe(filter_outlier)
    test = pd.read_csv('data/FuSai.csv', encoding = 'gb18030').\
        rename({'样本id':'id','收率':'target'},axis=1)
    optimize = pd.read_csv('data/optimize.csv', encoding = 'gb18030').\
        rename({'样本id':'id','收率':'target'},axis=1)
    
    testA = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding = 'gb18030').\
        rename({'样本id':'id','收率':'target'},axis=1)
    testB = pd.read_csv('data/jinnan_round1_testB_20190121.csv', encoding = 'gb18030').\
        rename({'样本id':'id','收率':'target'},axis=1)
    testC = pd.read_csv('data/jinnan_round1_test_20190201.csv', encoding = 'gb18030').\
        rename({'样本id':'id','收率':'target'},axis=1)
    
    ansA = pd.read_csv('data/jinnan_round1_ansA_20190125.csv', encoding = 'gb18030',header=None).\
        rename({0:'id',1:'target'},axis=1)
    ansB = pd.read_csv('data/jinnan_round1_ansB_20190125.csv', encoding = 'gb18030',header=None).\
        rename({0:'id',1:'target'},axis=1)
    ansC = pd.read_csv('data/jinnan_round1_ans_20190201.csv', encoding = 'gb18030',header=None).\
        rename({0:'id',1:'target'},axis=1)
    unique1_map,unique2_map,unique3_map = get_maps(train)
    
    
    
    A = pd.merge(testA,ansA,on='id',how='left').pipe(filter_outlier)
    B = pd.merge(testB,ansB,on='id',how='left').pipe(filter_outlier)
    C = pd.merge(testC,ansC,on='id',how='left').pipe(filter_outlier)
    
    train_data = pd.concat([train,A,B,C],axis=0,ignore_index=True)
    test_data = test
    
    return train_data,test_data,optimize
