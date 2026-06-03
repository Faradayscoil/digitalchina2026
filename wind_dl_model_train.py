import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras import regularizers
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import warnings
import random

matplotlib.rcParams['font.family'] = 'STSong'

warnings.filterwarnings('ignore')

filename1 = r'.\wind_split\4081950112845135880\wind_train.csv'
filename2 = r'.\wind_split\4081950112845135895\wind_train.csv'
filename3 = r'.\wind_split\4081950112845135971\wind_train.csv'
filename4 = r'.\wind_split\4081950112845135975\wind_train.csv'
filename5 = r'.\wind_split\4081950112845136015\wind_train.csv'
# 参数设置
seed = 2026
epsilon = 1
HISTORY_LEN = 96          # 历史窗口长度（15分钟*96 = 24小时）
FORECAST_LEN = 16         # 预测未来4小时（16个15分钟点）
TARGET_COL = '功率'        # 目标列名
BATCH_SIZE = 256
EPOCHS = 200
VALIDATION_SPLIT = 0.2     # 从训练集末尾划分验证集比例


# 数据加载与预处理函数
def load_and_preprocess(data_path, is_train=True):
    """
    加载CSV数据，处理缺失值，添加sin/cos风向特征，返回DataFrame。
    如果是训练数据，会返回包含目标列的数据；测试数据则不包含目标列。
    """
    df = pd.read_csv(data_path, parse_dates=['时间'])
    df.set_index('时间', inplace=True)

    # 提取时间特征，小时、天、月份、季节
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month

    # 季节定义：春季(3-5) -> 1, 夏季(6-8) -> 2, 秋季(9-11) -> 3, 冬季(12-2) -> 4
    def get_season(month):
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4
    df['season'] = df['month'].map(get_season)

    # 删除装机列
    if '装机' in df.columns:
        df.drop(columns=['装机'], inplace=True)

    # 功率异常值处理
    df.loc[df['功率'] < 0, '功率'] = 0

    # 风向角度转换为弧度（在缺失值处理之前）
    wind_dirs = ['10米风向', '30米风向', '50米风向', '70米风向', '轮毂高度风向']
    for col in wind_dirs:
        if col in df.columns:
            df[col] = np.deg2rad(df[col])  # 度 -> 弧度，覆盖原列

    # 缺失值处理：先用线性填充，再前向填充，剩余少量用后向填充
    df.interpolate(method='linear', limit=1, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # 二阶特征交叉
    def cross_features_make(input_data, func_dict, col_list):
        input_data = input_data.copy()
        n = len(col_list)
        for i in range(n):
            for j in range(i + 1, n):  # 仅当 i < j 时
                col_i = col_list[i]
                col_j = col_list[j]
                for func_name, func in func_dict.items():
                    # 生成特征时，对于除法，我们只生成 col_i / col_j 一种方向
                    func_features = func(input_data[col_i], input_data[col_j])
                    col_func_features = '-'.join([col_i, func_name, col_j])
                    input_data[col_func_features] = func_features
        return input_data

    # 选择参与交叉的特征
    exclude_cols = ['hour', 'day', 'month', 'season'] + wind_dirs + \
                   ['10m气温', '10m气压', '10m湿度', '功率']
    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    func_dict1 = {
        'div': lambda x, y: np.log(x + epsilon) - np.log(y + epsilon),
        'multi': lambda x, y: x * y
    }
    df = cross_features_make(df, func_dict1, candidate_cols)

    # 交叉特征中可能出现无穷值，用后向填充处理
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 将风向角度转换为sin/cos分量（0~360度）
    for col in wind_dirs:
        if col in df.columns:
            rad = df[col]               # 已经是弧度
            df[f'{col}_sin'] = np.sin(rad)
            df[f'{col}_cos'] = np.cos(rad)
            df.drop(columns=[col], inplace=True)

    # 将剩余的无穷值填充为0
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # 选择特征列（不包括时间索引和功率）
    feature_cols = [c for c in df.columns if c != TARGET_COL] if is_train else df.columns.tolist()

    # 确保所有特征数值类型
    df = df.astype(float)

    return df, feature_cols


def create_supervised_samples(df, feature_cols, history_len, forecast_len, target_col=None):
    """
    从DataFrame构建监督学习样本。
    X: (样本数, history_len, len(feature_cols))
    y: (样本数, forecast_len) 如果target_col不为None，否则返回None（用于测试）
    """
    data = df[feature_cols].values
    if target_col:
        targets = df[target_col].values
    else:
        targets = None

    X, y = [], []
    # 样本起始点：需要保证有完整的历史和未来数据
    start_idx = history_len
    end_idx = len(df) - forecast_len + 1 if target_col else len(df) - history_len + 1
    for i in range(start_idx, end_idx):
        X.append(data[i - history_len:i])
        if target_col:
            y.append(targets[i:i + forecast_len])

    if target_col:
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    else:
        return np.array(X, dtype=np.float32), None


def lstm_model(input_feature_dim):
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(HISTORY_LEN, input_feature_dim), dropout=0.4,
                   recurrent_dropout=0.4,
                   kernel_regularizer=regularizers.l2(1e-3),
                   return_sequences=False))
    # model.add(LSTM(units=64, dropout=0.3,
    #                recurrent_dropout=0.3,
    #                kernel_regularizer=regularizers.l2(1e-4),
    #                return_sequences=False))
    model.add(Dense(units=32, activation='relu',
                    kernel_regularizer=regularizers.l2(1e-3)
                    ))
    model.add(Dense(units=FORECAST_LEN))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model


if __name__ == '__main__':
    # 设置种子
    random.seed(seed)
    print(f"使用种子{seed}生成的随机数:", random.random())

    # 加载训练数据
    train_df, feature_cols = load_and_preprocess(filename1, is_train=True)
    print(f"训练数据形状: {train_df.shape}")
    print(f"特征列: {feature_cols}")

    # 特征缩放（基于训练集）
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 拟合特征缩放器
    X_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    # PCA降维
    pca = PCA(n_components=0.95)  # 保留95%方差，也可指定具体维数，如 n_components=50
    X_pca = pca.fit_transform(X_scaled)
    # # 替换原始数据中的特征为缩放后的值
    # train_df[feature_cols] = X_scaled

    if TARGET_COL in train_df.columns:
        y_scaled = scaler_y.fit_transform(train_df[[TARGET_COL]].values).flatten()
        train_df[TARGET_COL] = y_scaled
    else:
        raise ValueError("训练数据中缺少目标列 '功率'")

    # 将降维后的特征组成新的DataFrame
    pca_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, index=train_df.index, columns=pca_columns)
    # 添加目标列（如果有）
    if TARGET_COL in train_df.columns:
        df_pca['功率'] = train_df['功率'].values
    # 更新特征列列表
    new_feature_cols = pca_columns.copy()
    print(f"PCA降维后特征数: {len(new_feature_cols)}")

    # 构建监督学习样本
    # X_train_all, y_train_all = create_supervised_samples(
    #     train_df, feature_cols, HISTORY_LEN, FORECAST_LEN, TARGET_COL)
    X_train_all, y_train_all = create_supervised_samples(
        df_pca, new_feature_cols, HISTORY_LEN, FORECAST_LEN, TARGET_COL)
    print(f"总样本数: {X_train_all.shape[0]}")

    # 按时间顺序划分训练集和验证集（保持时序，最后VALIDATION_SPLIT部分作为验证）
    split_idx = int(X_train_all.shape[0] * (1 - VALIDATION_SPLIT))
    X_train, X_val = X_train_all[:split_idx], X_train_all[split_idx:]
    y_train, y_val = y_train_all[:split_idx], y_train_all[split_idx:]
    print(f"训练样本数: {X_train.shape[0]}, 验证样本数: {X_val.shape[0]}")

    # 回调
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(r'.\wind_results\best_model.h5', monitor='val_loss', save_best_only=True)
    ]

    model_lstm = lstm_model(len(new_feature_cols))
    # 训练模型
    history = model_lstm.fit(X_train, y_train,
                             batch_size=BATCH_SIZE, epochs=EPOCHS,
                             validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
