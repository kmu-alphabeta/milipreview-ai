---
# **Milipreview AI**
이 프로젝트의 AI 파트는 군 입대 지원자를 위한 합격 컷 예측과 사용자 맞춤형 합격 확률 계산을 담당합니다.

시계열 데이터를 처리하기 위해 LSTM(Long Short-Term Memory) 모델을 사용하며, 육군, 해군, 공군, 해병대 등 군별 맞춤형 모델을 학습합니다. 각 모델은 독립적으로 설계, 학습, 저장되어 높은 예측 정확도와 유지보수를 목표로 합니다.
---

## **프로젝트 개요**

이 프로젝트는 과거 군대 모집 데이터를 활용하여 미래의 **군 합격 컷**을 예측하고, 사용자가 입력한 점수와 지원 정보를 바탕으로 **합격 확률**을 반환하는 시스템을 제공합니다.

육군, 해군, 공군, 해병대 등 **군별 맞춤 모델**을 학습하며, 각각의 군 데이터에 따라 독립적으로 학습된 모델을 사용하여 미래의 합격 컷 점수를 예측합니다.

---

## **주요 기능**

1. **군별 합격 컷 예측**:

   - 각 군(육군, 해군, 공군, 해병대) 데이터를 학습하여 미래 특정 연도, 월, 모집 계열의 합격 컷 점수를 예측합니다.

2. **사용자 합격 확률 계산**:

   - 사용자가 입력한 연도, 월, 모집 계열, 점수를 기반으로 해당 군에서 합격할 확률을 반환합니다.

3. **군별 독립 모델**:
   - 각 군(Army, Navy, Air Force, Marine Corps)에 대해 별도의 모델 학습 및 예측 파일을 구성하여 군별 특성을 반영합니다.

---

## **폴더 구조**

```
MODEL/
├── air-force/                   # 공군
│   ├── trained_model/           # 공군 학습된 모델 저장
│   └── air-force_train_model.py # 공군 모델 학습 코드
├── army/                        # 육군
│   ├── trained_model/           # 육군 학습된 모델 저장
│   └── army_train_model.py      # 육군 모델 학습 코드
├── data/                        # 데이터 파일 폴더
│   └── (각 군의 데이터 파일을 여기에 저장)
├── marine/                      # 해병대
│   ├── trained_model/           # 해병대 학습된 모델 저장
│   └── marine_train_model.py    # 해병대 모델 학습 코드
├── navy/                        # 해군
│   ├── trained_model/           # 해군 학습된 모델 저장
│   ├── navy_train_model.py      # 해군 모델 학습 코드
│   └── navy_predict_model.py    # 해군 사용자 예측 코드
├── predict_model.py             # 통합 예측 코드
├── README.md
├── requirements.txt             # 필수 라이브러리
└── milipreview-navy.ipynb
```

---

## **기술 스택**

- **프로그래밍 언어**: Python
- **머신러닝**:
  - TensorFlow/Keras: 모델 설계 및 학습
  - Scikit-learn: 데이터 전처리 및 평가
- **데이터 처리**:
  - Pandas: 데이터 조작 및 분석
  - NumPy: 수치 계산
- **시각화**:
  - Matplotlib, Seaborn: 결과 시각화

---

## **설치 방법**

1. **필수 라이브러리 설치**

   ```bash
   pip install -r requirements.txt
   ```

2. **데이터 준비**
   - 데이터를 `data/` 폴더에 저장합니다.
   - 각 군의 데이터를 별도로 준비해야 하며, CSV 파일 형식이어야 합니다.
     ```csv
     year,month,category,score
     2022,1,일반,80
     2022,2,일반,82
     ```

---

## **사용 방법**

### 1. **군별 모델 학습**

- 각 군에 대해 독립적으로 모델을 학습합니다.

#### 육군(Army) 모델 학습

```bash
python army/army_train_model.py
```

#### 해군(Navy) 모델 학습

```bash
python navy/navy_train_model.py
```

#### 공군(Air Force) 모델 학습

```bash
python air-force/air-force_train_model.py
```

#### 해병대(Marine) 모델 학습

```bash
python marine/marine_train_model.py
```

---

## **예상 결과**

1. **모델 학습 로그**

   ```
   Epoch 100, Loss: 0.00234
   Epoch 200, Loss: 0.00187
   ...
   Training complete. Model saved to trained_model/
   ```

2. **합격 확률 계산 결과**
   ```
   군: 해군
   예측된 합격 컷: 82
   사용자의 점수: 85
   사용자의 합격 확률: 95%
   ```

---

## **향후 개선 사항**

- Bidirectional LSTM 및 Attention Mechanism 적용.
