import numpy as np
from math import exp
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU만 사용
from tensorflow.keras.models import load_model
import joblib

# 모델 로드
def load_category_model(military, category):
    try:
        model = load_model(f"{military}/trained_model/model_{category}.h5")
        scaler_X = joblib.load(f"{military}/trained_model/scaler_X_{category}.pkl")
        scaler_y = joblib.load(f"{military}/trained_model/scaler_y_{category}.pkl")
        return model, scaler_X, scaler_y
    except Exception as e:
        raise ValueError(f"{category} 카테고리에 대한 모델을 로드할 수 없습니다. 에러: {e}")

# 합격컷 예측 함수
def predict_cutoff(millitary, category, year, month):
    # 모델과 스케일러 로드
    model, scaler_X, scaler_y = load_category_model(millitary, category)

    time_index = year + month / 12
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    scaler_year = MinMaxScaler()
    scaler_year.fit(np.array([2022, 2023, 2024]).reshape(-1, 1))  # 학습 시 사용된 연도 범위
    year_scaled = scaler_year.transform([[year]])[0][0]
    year_weighted = year_scaled * 1.7  # 연도에 가중치 적용

    X_future = np.array([[time_index, sin_month, cos_month, year_scaled, year_weighted]])

    # 입력 데이터 정규화 및 LSTM 형식으로 변환
    X_future = scaler_X.transform(X_future)
    X_future = X_future.reshape((1, 1, X_future.shape[1]))

    # 예측 수행 및 정규화 해제
    predicted_cutoff = model.predict(X_future)
    predicted_cutoff = scaler_y.inverse_transform(predicted_cutoff)[0][0]
    return predicted_cutoff

# 확률 계산 함수 정의
def calculate_probability(user_input, predicted_cutoff):
    difference = user_input - predicted_cutoff

    if difference < 0:
        # 커트라인을 넘지 못한 경우
        return 1 / (1 + exp(-0.15 * difference))  # 낮은 확률
    elif 0 <= difference < 5:
        # 커트라인과 5점 이내
        return 0.7 + (0.05 * difference)  # 선형 증가
    else:
        # 커트라인을 5점 이상 초과
        return 0.9 + 0.01 * (difference - 5)  # 높은 확률

# 사용자 입력------------------------------- 
# 합격컷, 합불 예측 함수
def user_predict(military, category, year, month, user_score):
    if military == "army":
        category = category.replace('/', '+')
    predicted_cutoff = predict_cutoff(military, category, year, month)
    pass_probability = calculate_probability(user_score, predicted_cutoff)
    pass_probability = int(round(pass_probability, 2) * 100)
    if predicted_cutoff < user_score:
        pass_status = "합격"
    else: pass_status = "불합격"
    
    return predicted_cutoff, pass_probability, pass_status

if __name__ == "__main__":
    military = "army"   # 종류 : army, air-force, army, marine
    category = "K계열전차승무"
    user_input = 50
    year = 2025
    month = 2
    
    print(user_predict(military, category, year, month, user_input))

# pass_probability = calculate_probability(user_input, predict_cutoff(category, year, month))
# print(f"사용자가 {month}월에 합격할 확률: {pass_probability * 100:.2f}%")

# 2025년 월별 카테고리별 합격컷 예측
# print(f"=== 카테고리: {category} ===")
# for month in range(1, 13):
#     predicted_cutoff = predict_cutoff(category, 2025, month)
#     print(f"2025년 {month}월 예측 커트라인: {predicted_cutoff:.2f}")
