import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
import os
import tarfile

# 데이터 로드 및 전처리
df = pd.read_csv("data\\navy_data.csv")
df['time_index'] = df['year'] + df['month'] / 12
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

categories = df['category'].unique()

# 학습 진행 중 로그 출력 콜백
class CustomPrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {logs['loss']}")

# 카테고리별 모델 학습 및 저장
base_dir = "navy/trained_model"
os.makedirs(base_dir, exist_ok=True)

for category in categories:
    print(f"Processing category: {category}")
    
    # 데이터 필터링
    df_category = df[df['category'] == category].copy()
    scaler_year = MinMaxScaler()
    df_category['year_scaled'] = scaler_year.fit_transform(df_category['year'].values.reshape(-1, 1))
    df_category['year_weighted'] = df_category['year_scaled'] * 1.7

    X = df_category[['time_index', 'sin_month', 'cos_month', 'year_scaled', 'year_weighted']].values
    y = df_category['score'].values

    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    X = X.reshape((X.shape[0], 1, X.shape[1]))
    weights = np.exp(np.linspace(0, 3, len(X)))

    if len(X) > 1:  # 데이터 분할
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, shuffle=False
        )
    else:
        print(f"Category {category} has limited data. Using all data for training.")
        X_train, y_train, weights_train = X, y, weights
        X_test, y_test = X, y

    # LSTM 모델 정의
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(100, activation='relu', return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, sample_weight=weights_train, epochs=500, batch_size=16, verbose=0, callbacks=[CustomPrintCallback(), early_stopping])

    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"카테고리: {category}, Test MSE: {mse}, Test MAE: {mae}, R²: {r2}")

    # 모델 저장 디렉토리 생성
    version_dir = os.path.join(base_dir, f"model_{category}/1")
    os.makedirs(version_dir, exist_ok=True)

    # 모델 저장 (SavedModel 형식)
    model.save(version_dir)
    joblib.dump(scaler_X, os.path.join(version_dir, f"scaler_X_{category}.pkl"))
    joblib.dump(scaler_y, os.path.join(version_dir, f"scaler_y_{category}.pkl"))

    # .tar.gz로 압축
    tar_path = os.path.join(base_dir, f"model_{category}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(version_dir, arcname="1")  # 최상위 디렉토리 '1'로 압축

    print(f"Category {category}: Saved model and scalers to {tar_path}")

print("모든 카테고리에 대해 모델 학습 및 저장이 완료되었습니다.")
