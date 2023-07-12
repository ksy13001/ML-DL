import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np

# 선형 회귀 : 특성이 하나인 경우 어떤 직선을 학습 하는 알고리즘 (k-최근접 이웃 회귀 알고리즘은 샘플이 훈련 세트 범위를 벗어나면 제대로 된 예측이 어렵기에 사용)

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
# ------------------------- 훈련 세트, 테스트 세트 준비 -------------------------------------
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = np.reshape(train_input, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))

# -------------------------- 선형 회귀 모델 훈련 -------------------------------------------
w = 50
lr = LinearRegression()
lr.fit(train_input, train_target)
p = lr.predict([[w]])
print('predict weight : ', p)
# lr.coef_ : 기울기 / lr.intercept : y 절편 / 둘 다 ML 알고리즘이 찾은 값 -> 모델 파라미터(대부분의 ML 알고리즘은 최적의 모델 파라미터를 찾는 알고리즘)
print(lr.coef_, lr.intercept_)  # [39.017144] - 709.01864
print('----------------------1차 방정식---------------------')
print('훈련 세트 score : ', lr.score(train_input, train_target))
print('테스트 세트 score : ', lr.score(test_input, test_target))

# --------------------------- 2차 방정식 선형 회귀 모델 ------------------------------------------
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
lr.fit(train_poly, train_target)
p = lr.predict([[w**2, w]])
print(lr.coef_, lr.intercept_)
print('----------------------다항 회귀--------------------')
print('훈련 세트 score :', lr.score(train_poly, train_target))
print('테스트 세트 score :', lr.score(test_poly, test_target))
# --------------------------------- 그래프 ------------------------------------------------------
plt.scatter(train_input, train_target)
point = np.arange(15, w)
print('point :', point)
# plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.plot(point, 1.01*point**2 - 21.55*point + 116.05)
plt.scatter(w, p, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
