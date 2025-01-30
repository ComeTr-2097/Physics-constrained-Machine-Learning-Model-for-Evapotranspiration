# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:27:53 2024

@author: Chen Zhang
"""

import Hybrid_main2
import Hybrid_f2
from data_prep_M import X2_scaled, y2, EC_total

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization#, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import HeNormal

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


# 自定义损失函数
def custom_loss(y_true, y_pred):
    
    LE_true = y_true[:, 1]
    numerator = y_true[:, 2]
    DELTA = y_true[:, 3]
    gamma = y_true[:, 4]
    ra = y_true[:, 5]
    rs = tf.exp(y_pred[:, 0])
    # cp=1013
    LE_pred = numerator / (DELTA + gamma * (1 + rs / ra))
    
    # 计算损失
    loss = tf.reduce_mean(tf.square(LE_true - LE_pred))
    return loss


# 定义创建ANN模型的函数
def create_model(hidden_layers=6, neurons_per_layer=64):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', kernel_initializer=HeNormal(), input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())  # 添加BN层
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu', kernel_initializer=HeNormal()))
        # model.add(Dropout(rate = 0.5))  # 在隐藏层后添加Dropout层
        model.add(BatchNormalization())  # 添加BN层
    model.add(Dense(1, activation='relu'))  # 输出层
    optimizer = Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model

# 计算损失函数中的变量
external = pd.DataFrame()
external['es'] = Hybrid_f2.es_calc(EC_total['TA_Avg'].values)/100
external['ea'] = Hybrid_f2.ea_calc(EC_total['TA_Avg'].values,
                                   EC_total['RH_Avg'].values)/100
external['DELTA'] = Hybrid_f2.Delta_calc(EC_total['TA_Avg'].values)/100
external['gamma'] = Hybrid_f2.gamma_calc(EC_total['P_Avg'].values*100)/100
external['rho'] = Hybrid_f2.rho_calc(EC_total['TA_Avg'].values,
                                     EC_total['P_Avg'].values*100)
external['Rn'] = EC_total['Rn_Avg']
external['G'] = EC_total['G_Avg']
external['ra'] = EC_total['ra']

external['DELTA*(Rn-G)+rho*1013*(es-ea)/ra'] = external['DELTA']*(external['Rn']-external['G'])+external['rho']*1013*(external['es']-external['ea'])/external['ra']

external = external[['DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']]
external['LE'] = EC_total['LE']

# 准备训练集和验证集
external = external[['LE','DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']]
y2 = np.log(y2)
Xye = pd.concat([X2_scaled, y2, external], axis=1)

# 数据集划分为训练集和验证集
X = Xye[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']].values
y = Xye[['rs','LE','DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']].values
X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=1/3, random_state=42, shuffle=True)

# var = 'TA_Avg'

# var_1 = Xye[var].quantile(0.01)

# Xye_trainval = Xye[Xye[var]>var_1]
# X_trainval = Xye_trainval[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']].values
# y_trainval = Xye_trainval[['rs','LE','DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']].values

# # 数据集划分为训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.3, random_state=42, shuffle=True)


# # 测试集
# Xye_val1 = Xye[Xye[var]<=var_1]
# X_val1 = Xye_val1[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']].values
# y_val1 = Xye_val1[['rs','LE','DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']].values

# var_99 = Xye[var].quantile(0.99)

# Xye_trainval = Xye[Xye[var]<var_99]
# X_trainval = Xye_trainval[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']].values
# y_trainval = Xye_trainval[['rs','LE','DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']].values

# # 数据集划分为训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.3, random_state=42, shuffle=True)

# Xye_val99 = Xye[Xye[var]>=var_99]
# X_val99 = Xye_val99[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']].values
# y_val99 = Xye_val99[['rs','LE','DELTA*(Rn-G)+rho*1013*(es-ea)/ra','DELTA','gamma','ra']].values

# 设置batch_size
batch_size = 32

# 定义回调函数
checkpoint_filepath = 'Hybrid_checkpoint.h5'
model_check = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True, #只保存模型权重 (保存整个模型:False)
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

log = CSVLogger('Hybrid_training.log')

# # 设置ReduceLROnPlateau回调
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.001, verbose=1)

# 设置EarlyStopping回调
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)

# 创建模型
model = create_model()
# 训练模型
history = model.fit(
    X_train, y_train, 
    epochs=1000, 
    batch_size=batch_size, 
    verbose=1, 
    validation_data=(X_val, y_val),
    callbacks=[model_check, log, early_stopping] # reduce_lr
)

# 加载最佳模型权重
model.load_weights(checkpoint_filepath)

# from tensorflow.keras.models import load_model
# # 加载模型
# loaded_model = load_model(checkpoint_filepath, custom_objects={'custom_loss': custom_loss})

# 读取训练日志
training_log = pd.read_csv('Hybrid_training.log')

# 绘制loss曲线
plt.plot(training_log['loss'], label='Training Loss')
plt.plot(training_log['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
# 设置x轴坐标范围
plt.xlim(0, 1000)
# 设置x轴刻度大小
plt.xticks(np.arange(0, 1000, 100))
plt.ylabel('Loss')
# 设置y轴坐标范围
plt.ylim(0, 1000)
# 设置y轴刻度大小
plt.yticks(np.arange(0, 1000, 100))
plt.legend()
plt.show()


# 训练集和验证集性能
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
# y_val1_pred = np.squeeze(model.predict(X_val1), axis=1)
# y_val99_pred = np.squeeze(model.predict(X_val99), axis=1)

# plt.scatter(y_train[:,0], y_train_pred)
# plt.scatter(y_val[:,0], y_val_pred)
# plt.scatter(y_val1[:,0], y_val1_pred)
# plt.scatter(y_val99[:,0], y_val99_pred)


LE_train_true = y_train[:, 1]
numerator_train = y_train[:, 2]
DELTA_train = y_train[:, 3]
gamma_train = y_train[:, 4]
ra_train = y_train[:, 5]
rs_train = np.exp(y_train_pred[:, 0])
# cp=1013
LE_train_pred = numerator_train / (DELTA_train + gamma_train * (1 + rs_train / ra_train))


LE_val_true = y_val[:, 1]
numerator_val = y_val[:, 2]
DELTA_val = y_val[:, 3]
gamma_val = y_val[:, 4]
ra_val = y_val[:, 5]
rs_val = np.exp(y_val_pred[:, 0])
# cp=1013
LE_val_pred = numerator_val / (DELTA_val + gamma_val * (1 + rs_val / ra_val))


LE_test_true = y_test[:, 1]
numerator_test = y_test[:, 2]
DELTA_test = y_test[:, 3]
gamma_test = y_test[:, 4]
ra_test = y_test[:, 5]
rs_test = np.exp(y_test_pred[:, 0])
# cp=1013
LE_test_pred = numerator_test / (DELTA_test + gamma_test * (1 + rs_test / ra_test))


# LE_true_val1 = y_val1[:, 1]
# numerator_val1 = y_val1[:, 2]
# DELTA_val1 = y_val1[:, 3]
# gamma_val1 = y_val1[:, 4]
# ra_val1 = y_val1[:, 5]
# rs_val1 = y_val1_pred
# # cp=1013
# LE_pred_val1 = numerator_val1 / (DELTA_val1 + gamma_val1 * (1 + rs_val1 / ra_val1))


# LE_true_val99 = y_val99[:, 1]
# numerator_val99 = y_val99[:, 2]
# DELTA_val99 = y_val99[:, 3]
# gamma_val99 = y_val99[:, 4]
# ra_val99 = y_val99[:, 5]
# rs_val99 = y_val99_pred[:, 0]
# cp=1013
# LE_pred_val99 = numerator_val99/ (DELTA_val99 + gamma_val99 * (1 + rs_val99 / ra_val99))


# plt.scatter(LE_true_train,LE_pred_train)
# plt.scatter(LE_true_val,LE_pred_val)
# plt.scatter(LE_true_val1,LE_pred_val1)
# plt.scatter(LE_true_val99,LE_pred_val99)


print('r2_train:',r2_score(LE_train_true, LE_train_pred))
print('mse_train:',mean_squared_error(LE_train_true, LE_train_pred))
print('mae_train:',mean_absolute_error(LE_train_true, LE_train_pred))

print('r2_val:',r2_score(LE_val_true, LE_val_pred))
print('mse_val:',mean_squared_error(LE_val_true, LE_val_pred))
print('mae_val:',mean_absolute_error(LE_val_true, LE_val_pred))

print('r2_test:',r2_score(LE_test_true, LE_test_pred))
print('mse_test:',mean_squared_error(LE_test_true, LE_test_pred))
print('mae_test:',mean_absolute_error(LE_test_true, LE_test_pred))

# print('r2_val1:',r2_score(LE_true_val1, LE_pred_val1))
# print('mse_val1:',mean_squared_error(LE_true_val1, LE_pred_val1))
# print('mae_val1:',mean_absolute_error(LE_true_val1, LE_pred_val1))

# print('r2_val99:',r2_score(LE_true_val99, LE_pred_val99))
# print('mse_val99:',mean_squared_error(LE_true_val99, LE_pred_val99))
# print('mae_val99:',mean_absolute_error(LE_true_val99, LE_pred_val99))




