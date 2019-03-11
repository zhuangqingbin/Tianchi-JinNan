#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd

def filter_outlier(data):
    df = data.copy()
    return df[(df.target<=1) & (df.target>=0.85) & (df.B14>=350) & (df.B14<=460)].reset_index(drop=True)


def get_maps(train_data):
    train_df = train_data.copy()
    regulation_map = train_df.groupby('B14')['target'].nunique()
    
    unique1_map = {}
    for i in regulation_map[regulation_map==1].index:
        unique1_map[i] = train_df[train_df.B14==i].target.iloc[0]
    
    unique2_map = {}
    for i in regulation_map[regulation_map==2].index:
        unique2_map[i] = train_df[train_df.B14==i].target.value_counts().index.tolist()
    
    unique3_map = {}
    for i in regulation_map[regulation_map==3].index:
        unique3_map[i] = train_df[train_df.B14==i].target.value_counts().index.tolist()
    
    return unique1_map,unique2_map,unique3_map
    
def map_by_regulation(data,unique1_map,unique2_map,unique3_map):
    df = data.copy()
    for key,value in enumerate(unique1_map):
        df.loc[df.B14==key,'value'] = value
    
    for key,value in enumerate(unique2_map):
        mean_value = np.mean(np.array(value))
        df.loc[df.B14==key,'value'] = df.loc[df.B14==key,'value'].apply(lambda x:value[0] if x>mean_value else value[1])
    
    for key,value in enumerate(unique3_map):
        df.loc[df.B14==key,'value'] = df.loc[df.B14==key,'value'].apply(lambda x:value[0] if abs(x-value[0])<=0.005 else x)
    
    return df
