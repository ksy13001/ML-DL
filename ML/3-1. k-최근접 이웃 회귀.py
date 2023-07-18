import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# k-최근접 이웃 회귀 알고리즘 : k-최근접 이웃 알고리즘을 사용한 회귀 문제로, 가장 가까운 이웃 샘플을 찾고 샘플들의 타깃값을 평균하여 예측하는 알고리즘

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
                         21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
                         23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
                         27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
                         39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
                         44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])

# ----------------훈련 세트, 테스트 세트 준비--------------------------
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
# reshape -> 배열의 크기 지정(-1은 나머지 원소로 채우라는 뜻)/ train, test -> 1차원 이므로 2차원으로 변경
train_input = np.reshape(train_input, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))

# ---------------------회귀모델 훈련------------------------------------
# 회귀모델 생성
knr = KNeighborsRegressor()
knr.n_neighbors = 3
knr.fit(train_input, train_target)
# 결정계수 R^2 출력 / 타깃이 평균을 예측할 수록 0 에 가까움 / 예측이 타깃에 가까울 수록 1에 가까움, 1에 가까울 수록 좋음
print(knr.score(test_input, test_target))
test_prediction = knr.predict(test_input)
# 타깃과 예측한 값 사이의 차이(타깃과 예측의 절댓값 오차의 평균)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)    # mae = 19.1571..., 19g 만큼 예측이 타깃과 차이난다.
# 이번엔 훈련 세트로 score
print(knr.score(train_input, train_target))
"""
테스트 세트의 결정계수 : 0.992809406101064
훈련 세트의 결정계수 : 0.9698823289099254
if 테스트 세트 점수 > 훈련 세트 점수 -> 과소적합 : 모델이 너무 단순하여 훈련세트에 적절히 훈련되지 않음
if 훈련 세트 점수 > 테스트 세트 점수 -> 과대적합 : 훈련 세트에만 잘 맞는 모델이라 새로운 샘플에 대한 예측을 만들때 잘 동작하지 않을 것임
과소적합 -> 모델을 더 복잡하게 만든다. - K 최근접 이웃 알고리즘 : 이웃의 개수 K 를 줄인다, knr.n_neighbors = 3
과대접합 -> 모델을 덜 복잡하게 만든다. 이웃의 개수 k 를 늘린다.
"""


# ----------------------그래프------------------------------------------
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
