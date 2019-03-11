#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import re
# 时间段
def getDuration(duration):
    try:
        hour_start,minute_start,hour_end,minute_end = re.findall(r"\d+\.?\d*",duration)
        time_interval = int(hour_end)-int(hour_start)+(int(minute_end)-int(minute_start))/60
        time_interval = time_interval+24 if time_interval<0 else time_interval
        return time_interval
    except:
        return np.nan


def process_data(tr,tr,op):
    train_data,test_data,optimize = tr.copy(),tr.copy(),op.copy()
    #A2和A3修正
    for df in [train_data,test_data,optimize]:
        df.A3[df.A3.isnull()] = df.A2[df.A3.isnull()]
        df.drop('A2',axis=1,inplace=True)
        
        
    # 时间戳
    time_cols = ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']
    for col in time_cols:
        for df in [train_data,test_data,optimize]:
            df[col] = pd.to_datetime(df[col]).apply(lambda t: (3600*t.hour+60*t.minute+t.second)/3600)
    
    
    # 时间间隔
    time_intervals = []
    for col_start,col_end in zip(time_cols[:-1],time_cols[1:]):
        time_intervals.append(f'{col_start}-{col_end}')
        for df in [train_data,test_data,optimize]:
            df[f'{col_start}-{col_end}'] = df[col_end]-df[col_start]
            df[f'{col_start}-{col_end}'] = df[f'{col_start}-{col_end}'].apply(lambda x:24+x if x<0 else x)
            


    time_duration_cols = ['A20','A28','B4','B9','B10','B11']
    for col in time_duration_cols:
        for df in [train_data,test_data,optimize]:
            df[col] = df[col].apply(getDuration)


    # 删除类别唯一的特征
    unique_sets = set()
    for df in [train_data,test_data]:
        for col in df.columns:
            if df[col].nunique() == 1:
                unique_sets.add(col)
    for df in [train_data,test_data,optimize]:
        df.drop(list(unique_sets), axis=1, inplace=True)
        
    # 删除缺失率超过90%的列
    na_sets = set()
    for df in [train_data,test_data]:
        for col in df.columns:
            if df[col].isnull().mean() > 0.9:
                na_sets.add(col)
    
    for df in [train_data,test_data,optimize]:
        df.drop(list(na_sets), axis=1, inplace=True)


    # 构建特征
    for df in [train_data,test_data,optimize]:
        df['A25'] = df['A25'].astype(np.float)
        df['A3_per'] = df['A3']/df['B14']
        df['B1_per'] = df['B1']*df['B2']/df['B14']
        df['B12_per'] = df['B12']/(df['B12']+df['B14'])
        df['b14/a1_a3_a4_a19_b1_b12'] = df['B14']/(df['A3']+df['A4']+df['A19']+df['B1']+df['B12'])
        df['A19A17'] = df['A19'] - df['A17']
        df['temp_sum'] = df['A10']+df['A12']+df['A15']+df['A17']+df['A21']+df['A25']+df['A27']+df['B6']+df['B8']
        df['time_sum'] = df['A9-A11']+df['A14-A16']+df['A16-A24']+df['A24-A26']+df['A26-B5']+df['B5-B7']
    all_cols = train_data.columns
    
    # 重新提取train、test
    train_data['intTarget'] = pd.cut(train_data['target'], 5, labels=False)
    train_data = pd.get_dummies(train_data, columns=['intTarget'])
    
    li = ['intTarget_0','intTarget_1','intTarget_2','intTarget_3','intTarget_4']
    
    mean_features = []
    for f1 in [col for col in all_cols if col not in ['id','target']]:
        cate_rate = train_data[f1].isnull().mean()
        if cate_rate < 0.50:
            for f2 in li:
                for f3 in ['B9','B10','B11','B12','B13','B14']:
                    col_name = f'{f3}_to_{f1}_{f2}_mean'
                    mean_features.append(col_name)
                    order_label = train_data.groupby([f1])[f2].mean()
                    train_data[col_name] = train_data[f3].map(order_label)
                    miss_rate = train_data[col_name].isnull().mean()
                    if miss_rate > 0.5:
                        train_data = train_data.drop([col_name], axis=1)
                        mean_features.remove(col_name)
                    else:
                        test_data[col_name] = test_data[f3].map(order_label)
                        optimize[col_name] = optimize[f3].map(order_label)
    train_data.drop(li, axis=1, inplace=True)
    
    mean_features = []
    for f in [col for col in all_cols if col not in ['id','target']]:
        col_name = f+"_target_mean"
        mean_features.append(col_name)
        target_label = train_data.groupby([f])['target'].mean()
        for df in [train_data,test_data,optimize]:
            df[col_name] = df[f].map(target_label) 
            
    target = train_data['target']
    train_data.drop(['target','id'], axis=1, inplace=True)
    test_data = test_data[train_data.columns]
    optimize_data = optimize[train_data.columns]
    
    return train_data,test_data,optimize_data
