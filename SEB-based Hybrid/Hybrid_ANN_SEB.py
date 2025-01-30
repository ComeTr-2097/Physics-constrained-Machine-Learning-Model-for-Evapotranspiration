# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:09:17 2024

@author: Chen Zhang(12214067@zju.edu.cn)
"""

import Hybrid_main_SEB
import Hybrid_f_SEB
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
    Rn = y_true[:, 2]
    G = y_true[:, 3]
    numerator = y_true[:, 4]
    ra = tf.exp(y_pred[:, 0])
    # cp=1013
    LE_pred = Rn - G - (numerator / ra)
    
    
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
    model.add(Dense(1, activation='softplus'))  # 输出层
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model

# 计算损失函数中的变量
external = pd.DataFrame()
external['TS'] = EC_total['TS_Avg']
external['TA'] = EC_total['TA_Avg']
external['rho'] = Hybrid_f2_SEB.rho_calc(EC_total['TA_Avg'].values,
                                     EC_total['P_Avg'].values*100)
external['Rn'] = EC_total['Rn_Avg']
external['G'] = EC_total['G_Avg']

external['rho*1013*(ts-ta)'] = external['rho']*1013*(external['TS']-external['TA'])

external = external[['Rn','G','rho*1013*(ts-ta)']]
external['LE'] = EC_total['LE']

# 准备训练集和验证集
external = external[['LE','Rn','G','rho*1013*(ts-ta)']]
y2 = np.log(y2)
Xye = pd.concat([X2_scaled,y2,external], axis=1)

# '''Check: True'''
# Xye['LE_calc'] = Xye['Rn'] - Xye['G'] - (Xye['rho*1013*(ts-ta)'] / Xye['ra'])

# 数据集划分为训练集和验证集
X = Xye[['TA_Avg','RH_Avg','P_Avg','Rain_Tot','WS_Avg','TS_Avg','SWC_Avg','NDVI','h']].values
y = Xye[['ra','LE','Rn','G','rho*1013*(ts-ta)']].values
X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=1/3, random_state=42, shuffle=True)


# 设置batch_size
batch_size = 32

# 定义回调函数
checkpoint_filepath = 'Hybrid2_checkpoint.h5'
model_check = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True, #只保存模型权重 (保存整个模型:False)
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

log = CSVLogger('Hybrid2_training.log')

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
    callbacks=[model_check, log, early_stopping]# ,reduce_lr
)

# 加载最佳模型权重
model.load_weights(checkpoint_filepath)

# from tensorflow.keras.models import load_model
# # 加载模型
# loaded_model = load_model(checkpoint_filepath, custom_objects={'custom_loss': custom_loss})

# 读取训练日志
training_log = pd.read_csv('Hybrid2_training.log')

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
plt.ylim(0, 5000)
# 设置y轴刻度大小
plt.yticks(np.arange(0, 5000, 500))
plt.legend()
plt.show()


# 训练集和验证集性能
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


LE_train_true = y_train[:, 1]
Rn_train = y_train[:, 2]
G_train = y_train[:, 3]
numerator_train = y_train[:, 4]
ra_train = np.exp(y_train_pred[:, 0])
# cp=1013
LE_train_pred = Rn_train - G_train - numerator_train / ra_train


LE_val_true = y_val[:, 1]
Rn_val = y_val[:, 2]
G_val = y_val[:, 3]
numerator_val = y_val[:, 4]
ra_val = np.exp(y_val_pred[:, 0])
# cp=1013
LE_val_pred = Rn_val - G_val - numerator_val / ra_val


LE_test_true = y_test[:, 1]
Rn_test = y_test[:, 2]
G_test = y_test[:, 3]
numerator_test = y_test[:, 4]
ra_test = np.exp(y_test_pred[:, 0])
# cp=1013
LE_test_pred = Rn_test - G_test - numerator_test / ra_test


print('r2_train:',r2_score(LE_train_true, LE_train_pred))
print('mse_train:',mean_squared_error(LE_train_true, LE_train_pred))
print('mae_train:',mean_absolute_error(LE_train_true, LE_train_pred))

print('r2_val:',r2_score(LE_val_true, LE_val_pred))
print('mse_val:',mean_squared_error(LE_val_true, LE_val_pred))
print('mae_val:',mean_absolute_error(LE_val_true, LE_val_pred))

print('r2_test:',r2_score(LE_test_true, LE_test_pred))
print('mse_test:',mean_squared_error(LE_test_true, LE_test_pred))
print('mae_test:',mean_absolute_error(LE_test_true, LE_test_pred))

