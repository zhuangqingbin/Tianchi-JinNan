#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 

def modify_train(data):
    train_data = data.copy()
    # 异常点修正
    train_data['A5'] = train_data['A5'].replace('1900/1/21 0:00','21:00:00')
    train_data['A5'] = train_data['A5'].replace('1900/1/29 0:00','14:00:00')
    train_data['A9'] = train_data['A9'].replace('1900/1/9 7:00','23:00:00')
    train_data['A9'] = train_data['A9'].replace('700','7:00:00')
    train_data['A11'] = train_data['A11'].replace(':30:00','00:30:00')
    train_data['A11'] = train_data['A11'].replace('1900/1/1 2:30','21:30:00')
    train_data['A16'] = train_data['A16'].replace('1900/1/12 0:00','12:00:00')
    train_data['A20'] = train_data['A20'].replace('6:00-6:30分','6:00-6:30')
    train_data['A20'] = train_data['A20'].replace('18:30-15:00','14:30-15:00')
    train_data['A22'] = train_data['A22'].replace(3.5,9)
    train_data['A25'] = train_data['A25'].replace('1900/3/10 0:00',np.nan)
    train_data['A26'] = train_data['A26'].replace('1900/3/13 0:00','13:00:00')
    train_data['A28'] = train_data['A28'].replace('17:40-16:10','17:40-18:10')
    
    train_data['B1'] = train_data['B1'].replace(3.5,np.nan)
    train_data['B4'] = train_data['B4'].replace('15:00-1600','15:00-16:00')
    train_data['B4'] = train_data['B4'].replace('18:00-17:00','16:00-17:00')
    train_data['B4'] = train_data['B4'].replace('19:-20:05','19:05-20:05')
    train_data['B9'] = train_data['B9'].replace('23:00-7:30','23:00-00:30')
    
    train_data.loc[train_data['id'] == 'sample_1894','A5'] = '14:00:00'
    train_data.loc[train_data['id'] == 'sample_1234','A9'] = '0:00:00'
    train_data.loc[train_data['id'] == 'sample_1020','A9'] = '18:30:00'
    
    train_data.loc[train_data['id'] == 'sample_844','A11'] = '10:00:00'
    train_data.loc[train_data['id'] == 'sample_1348','A11'] = '17:00:00'
    train_data.loc[train_data['id'] == 'sample_25','A11'] = '00:30:00'
    train_data.loc[train_data['id'] == 'sample_1105', 'A11'] = '4:00:00'
    
    train_data.loc[train_data['id'] == 'sample_313', 'A11'] = '15:30:00'
    train_data.loc[train_data['id'] == 'sample_291', 'A14'] = '19:30:00'
    train_data.loc[train_data['id'] == 'sample_1398', 'A16'] = '11:00:00'
    
    
    train_data.loc[train_data['id'] == 'sample_1177', 'A20'] = '19:00-20:00'
    train_data.loc[train_data['id'] == 'sample_71', 'A20'] = '16:20-16:50'
    train_data.loc[train_data['id'] == 'sample_14', 'A20'] = '18:00-18:30'
    train_data.loc[train_data['id'] == 'sample_69', 'A20'] = '6:10-6:50'
    train_data.loc[train_data['id'] == 'sample_1500', 'A20'] = '23:00-23:30'
    train_data.loc[train_data['id'] == 'sample_1524', 'A24'] = '15:00:00'
    train_data.loc[train_data['id'] == 'sample_1524', 'A26'] = '15:30:00'
    train_data.loc[train_data['id'] == 'sample_1046', 'A28'] = '1:00-18:30'
    
    train_data.loc[train_data['id'] == 'sample_1230', 'B5'] = '17:00:00'
    train_data.loc[train_data['id'] == 'sample_97', 'B7'] = '1:00:00'
    train_data.loc[train_data['id'] == 'sample_752', 'B9'] = '11:00-14:00'
    
    train_data.loc[train_data['id'] == 'sample_609','B11'] = '11:00-12:00'
    
    train_data.loc[train_data['id'] == 'sample_643','B11'] = '12:00-13:00'
    train_data.loc[train_data['id'] == 'sample_1164','B11'] = '5:00-6:00'
    train_data.loc[train_data.B14==40,'B14'] = 400
    train_data.loc[train_data['id'] == 'sample_919', 'A9'] = '19:50:00'
    
    train_data.loc[train_data.B14==785,'B14'] = 385
    train_data.loc[train_data.A19==310,'A19'] = 300
    train_data.loc[train_data.A19==700,'A19'] = 100
    
    train_data.loc[train_data['id'] == 'sample_566', 'A5'] = '18:00:00'
    train_data.loc[train_data['id'] == 'sample_40', 'A20'] = '5:00-5:30'
    train_data.loc[train_data['id'] == 'sample_531', 'B5'] = '1:00:00'
    
    train_data.loc[train_data['id'] == 'sample_1039', 'A16'] = '00:30:00'
    train_data.loc[train_data['id'] == 'sample_10', 'A16'] = '3:00:00'
    
    return train_data
    
