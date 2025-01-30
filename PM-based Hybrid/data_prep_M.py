# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:39:00 2024

@author: dell
"""

import os
import pandas as pd
import numpy as np
import Hybrid_f2, Hybrid_main2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, KFold
import random
from sklearn.utils import shuffle
import calendar
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler


'''----------EC----------'''

EC_path = r'F:\中国通量与气象站点植被指数\EC_73_v3'

# 获取文件夹中所有通量站文件的文件名列表
ec_files = [f for f in os.listdir(EC_path) if f.endswith('.csv')]

# 初始化一个空的 DataFrame 用于存储合并后的数据
EC_total = pd.DataFrame()

# 遍历所有 CSV 文件，并将它们合并到一个 DataFrame 中
for ec in ec_files:
    ec_path = os.path.join(EC_path, ec)
    flux = pd.read_csv(ec_path)
    
    # 选择存在的质量控制字段
    qa_flux = flux.filter(like='QA_').columns
    # 如果存在至少一个质量控制波段
    if len(qa_flux) > 0:
        if flux['G_Avg'].isna().all():
            flux['G_Avg'] = 0
            # 对相应的8个变量进行<0.5赋为np.nan
            flux.loc[flux['QA_TA_Avg']<0.5,'TA_Avg'] = np.nan
            flux.loc[flux['QA_RH_Avg']<0.5,'RH_Avg'] = np.nan
            flux.loc[flux['QA_P_Avg']<0.5,'P_Avg'] = np.nan
            flux.loc[flux['QA_Rain_Tot']<0.5,'Rain_Tot'] = np.nan
            flux.loc[flux['QA_WS_Avg']<0.5,'WS_Avg'] = np.nan
            flux.loc[flux['QA_TS_Avg']<0.5,'TS_Avg'] = np.nan
            flux.loc[flux['QA_SWC_Avg']<0.5,'SWC_Avg'] = np.nan
            flux.loc[flux['QA_Rn_Avg']<0.5,'Rn_Avg'] = np.nan
            flux.loc[flux['QA_LE']<0.5,'LE'] = np.nan
            flux.loc[flux['QA_H']<0.5,'H'] = np.nan
            flux.loc[(flux['LE']+flux['H'])/(flux['Rn_Avg']-flux['G_Avg'])<0.8,'LE'] = np.nan
        else:
            # 对相应的8个变量进行<0.5赋为np.nan
            flux.loc[flux['QA_TA_Avg']<0.5,'TA_Avg'] = np.nan
            flux.loc[flux['QA_RH_Avg']<0.5,'RH_Avg'] = np.nan
            flux.loc[flux['QA_P_Avg']<0.5,'P_Avg'] = np.nan
            flux.loc[flux['QA_Rain_Tot']<0.5,'Rain_Tot'] = np.nan
            flux.loc[flux['QA_WS_Avg']<0.5,'WS_Avg'] = np.nan
            flux.loc[flux['QA_TS_Avg']<0.5,'TS_Avg'] = np.nan
            flux.loc[flux['QA_SWC_Avg']<0.5,'SWC_Avg'] = np.nan
            flux.loc[flux['QA_G_Avg']<0.5,'G_Avg'] = np.nan
            flux.loc[flux['QA_Rn_Avg']<0.5,'Rn_Avg'] = np.nan
            flux.loc[flux['QA_LE']<0.5,'LE'] = np.nan
            flux.loc[flux['QA_H']<0.5,'H'] = np.nan
            flux.loc[(flux['LE']+flux['H'])/(flux['Rn_Avg']-flux['G_Avg'])<0.8,'LE'] = np.nan

    else:#天尺度不存在质量控制波段，而且G_Avg为空值,ChinaFlux站点的天尺度产品
        flux.loc[pd.isna(flux['G_Avg']), 'G_Avg'] = 0
        # flux.loc[(flux['LE']+flux['H'])/(flux['Rn_Avg']-flux['G_Avg'])<0.8,'LE'] = np.nan

    flux['TIMESTAMP'] = pd.to_datetime(flux['TIMESTAMP'])
    flux_M = flux[['TIMESTAMP',
                   'TA_Avg',
                   'RH_Avg',
                   'P_Avg',
                   'WS_Avg',
                   'TS_Avg',
                   'SWC_Avg',
                   'G_Avg',
                   'Rn_Avg',
                   'NDVI',
                   'LE',
                   'H',
                   'h',
                   'zm',
                   'zh'
                   ]].resample('1M', on='TIMESTAMP', closed = 'right', label = 'right').mean()
    flux_M.reset_index(inplace = True) #TIMESTAMP作为连接列键
    flux_rain_M = flux[['TIMESTAMP','Rain_Tot']].resample('1M', on='TIMESTAMP', closed = 'right', label = 'right').sum()
    flux_rain_M.reset_index(inplace=True)
    flux_M = flux_M.merge(flux_rain_M, on='TIMESTAMP', how='outer')
    
    flux.set_index('TIMESTAMP', inplace=True)
    flux_count = flux[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','G_Avg','Rn_Avg','NDVI','LE','H']].resample('M').count() #
    # 计算每个月的天数
    flux_count['days_in_month'] = flux_count.index.to_series().dt.days_in_month
    #计算数据质量字段
    qa_flux = pd.DataFrame()
    qa_flux['QA_TA_Avg'] = flux_count['TA_Avg']/flux_count['days_in_month']
    qa_flux['QA_RH_Avg'] = flux_count['RH_Avg']/flux_count['days_in_month']
    qa_flux['QA_P_Avg'] = flux_count['P_Avg']/flux_count['days_in_month']
    qa_flux['QA_Rain_Tot'] = flux_count['Rain_Tot']/flux_count['days_in_month']
    qa_flux['QA_WS_Avg'] = flux_count['WS_Avg']/flux_count['days_in_month']
    qa_flux['QA_TS_Avg'] = flux_count['TS_Avg']/flux_count['days_in_month']
    qa_flux['QA_SWC_Avg'] = flux_count['SWC_Avg']/flux_count['days_in_month']
    qa_flux['QA_G_Avg'] = flux_count['G_Avg']/flux_count['days_in_month']
    qa_flux['QA_Rn_Avg'] = flux_count['Rn_Avg']/flux_count['days_in_month']
    qa_flux['QA_NDVI'] = flux_count['NDVI']/flux_count['days_in_month']
    qa_flux['QA_LE'] = flux_count['LE']/flux_count['days_in_month']
    qa_flux['QA_H'] = flux_count['H']/flux_count['days_in_month']
    qa_flux.reset_index(inplace=True)
    flux_M = flux_M.merge(qa_flux, on='TIMESTAMP', how='outer')
    
    flux_M['Year'] = flux_M['TIMESTAMP'].map(lambda x: int(str(x).split('-')[0]))
    flux_M['Month'] = flux_M['TIMESTAMP'].map(lambda x: int(str(x).split('-')[1]))
    flux_M['Label'] = flux_M['Year'].astype(str) + flux_M['Month'].astype(str)
    flux_M['StationId'] = flux['No'].unique()[0]
    flux_M['Name'] = flux['Station'].unique()[0]
    flux_M['Type'] = flux['Type'].unique()[0]
    flux_M = flux_M[['TIMESTAMP','Year','Month','Label',
                     'StationId','Name','Type',
                     'TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','G_Avg','Rn_Avg','NDVI','LE','H','h','zm','zh',
                     'QA_TA_Avg','QA_RH_Avg','QA_P_Avg','QA_Rain_Tot','QA_WS_Avg','QA_TS_Avg','QA_SWC_Avg','QA_G_Avg','QA_Rn_Avg','QA_NDVI','QA_LE','QA_H']] #

    EC_total = pd.concat([EC_total, flux_M], ignore_index=True)


EC_total = EC_total[EC_total['QA_TA_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_RH_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_P_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_Rain_Tot']>=0.5]
EC_total = EC_total[EC_total['QA_WS_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_TS_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_SWC_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_G_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_Rn_Avg']>=0.5]
EC_total = EC_total[EC_total['QA_LE']>=0.5]
EC_total = EC_total[EC_total['QA_H']>=0.5]


'''月尺度土壤热通量插补'''
g_data = pd.read_excel(r'F:\中国通量与气象站点植被指数\EC_73_G_M_2000_2022.xlsx')
g_data = g_data[['No_','date','surface_soil_heat_flux_sum']]
g_data['Year'] = g_data['date'].dt.year
g_data['Month'] = g_data['date'].dt.month
g_data['d'] = g_data.apply(lambda row: calendar.monthrange(row['Year'], row['Month'])[1], axis=1)
g_data['G_era'] = g_data['surface_soil_heat_flux_sum']/(g_data['d']*24*3600)
g_data.rename(columns={'No_': 'StationId'},inplace='True')
g_data = g_data[['StationId','Year','Month','G_era']]
EC_total = pd.merge(EC_total, g_data, how='left', on=['StationId','Year','Month'])

EC_totalgn0 = EC_total[EC_total['G_Avg']!=0]
EC_totalgn0 = EC_totalgn0.dropna(subset=['G_era'])
# plt.scatter(EC_totalgn0['G_Avg'],EC_totalgn0['G_era'])
slope, intercept, r_value, p_value, std_err = linregress(EC_totalgn0['G_era'], EC_totalgn0['G_Avg'])
# EC_totalgn0['G_p'] = EC_totalgn0['G_era']*slope+intercept
# plt.scatter(EC_totalgn0['G_Avg'],EC_totalgn0['G_p'])
# plt.xlim(-40, 40)
# plt.ylim(-40, 40)
EC_total.loc[EC_total['G_Avg'] == 0, 'G_Avg'] = EC_total['G_era']* slope + intercept

# EC_total = EC_total[EC_total['LE']>0]
# EC_total = EC_total[EC_total['H']>0]
# EC_total = EC_total[EC_total['Rn_Avg']>0]
# EC_total = EC_total[EC_total['G_Avg']>0]
# EC_total = EC_total[(EC_total['LE']+EC_total['H'])/(EC_total['Rn_Avg']-EC_total['G_Avg'])>=0.8]
# EC_total['LE_Corr'] = (EC_total['Rn_Avg']-EC_total['G_Avg'])/(1+(EC_total['H']/EC_total['LE']))

EC_total['VPD_Avg'] = Hybrid_f2.vpd_calc(EC_total['TA_Avg'].values,EC_total['RH_Avg'].values)/100 #from pa to hpa

EC_total['ra'] = Hybrid_f2.ra_calc(EC_total['h'].values,
                                    EC_total['zm'].values,
                                    EC_total['zh'].values,
                                    EC_total['WS_Avg'].values)
EC_total.loc[(EC_total['ra']<0)|(EC_total['ra']>500),'ra'] = np.nan

EC_total['rs'] = Hybrid_f2.rs_PM_Inv(EC_total['TA_Avg'].values,
                                    EC_total['RH_Avg'].values,
                                    EC_total['P_Avg'].values,
                                    EC_total['Rn_Avg'].values,
                                    EC_total['G_Avg'].values,
                                    EC_total['LE'].values,
                                    EC_total['h'].values,
                                    EC_total['zm'].values,
                                    EC_total['zh'].values,
                                    EC_total['WS_Avg'].values)
EC_total.loc[(EC_total['rs']<0)|(EC_total['rs']>2000),'rs'] = np.nan
# EC_total['ln(rs)'] = np.log(EC_total['rs'])
# EC_total.loc[EC_total['ln(rs)']<0,'ln(rs)'] = np.nan


EC_total = EC_total[['TIMESTAMP','Year','Month','Label',
                     'StationId','Name','Type',
                     'TA_Avg','RH_Avg','VPD_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','G_Avg','Rn_Avg','NDVI',
                     'LE','H',
                     'h','zm','zh',
                     'ra','rs']]
EC_total = EC_total.dropna(how='any')
EC_total.reset_index(drop=True, inplace = True)
stas = EC_total['Name'].value_counts()
# # 打乱数据集
# EC_total = shuffle(EC_total, random_state=42)


# 准备ANN模型数据
X = EC_total[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','G_Avg','Rn_Avg','NDVI','h']]
y = EC_total['LE']

# # 打乱数据集
# X_shuffle, y_shuffle = shuffle(X, y, random_state=42)

#对特征进行Z-score标准化
scaler_X = StandardScaler()

# X_shuffle_scaled = pd.DataFrame(scaler.fit_transform(X_shuffle), columns=X_shuffle.columns)

X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)


# 准备Hybrid模型数据
X2 = EC_total[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']]
y2 = EC_total['rs']

#对特征进行Z-score标准化
scaler_X2 = StandardScaler()
# # 打乱数据集
# X2_shuffle, y2_shuffle = shuffle(X2, y2, random_state=42)

# #对特征进行Z-score标准化
# X2_shuffle_scaled = pd.DataFrame(scaler_X2.fit_transform(X2_shuffle), columns=X2_shuffle.columns)

X2_scaled = pd.DataFrame(scaler_X2.fit_transform(X2), columns=X2.columns)

# # 检查结果是否一致
# y2_restored = pd.Series(scaler_y2.inverse_transform(y2_scaled.values.reshape(-1, 1)).flatten())
# y2_restored2 = y2_scaled* scaler_y2.scale_ + scaler_y2.mean_