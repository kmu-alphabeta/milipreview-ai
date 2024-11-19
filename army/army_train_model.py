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

df = pd.read_csv("data\\army_data.csv")
df['time_index'] = df['year'] + df['month'] / 12
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

categories = df['category'].unique()  # 카테고리별로 모델 학습

class CustomPrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {logs['loss']}")

# 카테고리별로 모델 학습 및 저장
for category in categories:
    df_category = df[df['category'] == category].copy()

    # year 특성에 가중치를 곱해 추가 
    scaler_year = MinMaxScaler()
    df_category['year_scaled'] = scaler_year.fit_transform(df_category['year'].values.reshape(-1, 1))
    df_category['year_weighted'] = df_category['year_scaled'] * 1.5  # 가중치 1.7배

    X = df_category[['time_index', 'sin_month', 'cos_month', 'year_scaled', 'year_weighted']].values
    y = df_category['score'].values

    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    # LSTM 입력 형식에 맞게 변환
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # 지수적 가중치 부여
    weights = np.exp(np.linspace(0, 3, len(X)))

    if len(X) > 1:  # 최소 두 개의 데이터가 있어야 분할 가능
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, shuffle=False
        )
    else:
        # 데이터가 부족한 경우 테스트 데이터를 사용하지 않음
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
    model.compile(optimizer=Adam(learning_rate=0.0007), loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, sample_weight=weights_train, epochs=800, batch_size=16, verbose=0, callbacks=[CustomPrintCallback(), early_stopping])

    # 테스트 데이터 평가
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)  # 예측 값 스케일 복원
    y_test = scaler_y.inverse_transform(y_test)  # 실제 값 스케일 복원

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"카테고리: {category}, Test MSE: {mse}, Test MAE: {mae}, R²: {r2}")

    # 학습된 모델과 스케일러 저장
    save_dir = "army/trained_model"
    
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # '/'를 '_'로 대체하거나 제거
    safe_category = category.replace('/', '+')
        
    model.save(os.path.join(save_dir, f"model_{safe_category}.h5"))  # 모델 저장
    joblib.dump(scaler_X, os.path.join(save_dir, f"scaler_X_{safe_category}.pkl"))  # 입력 스케일러 저장
    joblib.dump(scaler_y, os.path.join(save_dir, f"scaler_y_{safe_category}.pkl"))  # 출력 스케일러 저장

print("모든 카테고리에 대해 모델 학습 및 저장이 완료되었습니다.")