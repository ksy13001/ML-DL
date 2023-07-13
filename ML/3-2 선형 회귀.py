import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# ------------------------- 훈련 세트, 테스트 세트 준비 -----------------------------------------
train_input , test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = np.reshape(train_input, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))

# -------------------------- 선형 회귀 모델 훈련 ------------------------------------------------
lr = LinearRegression()
lr.fit(train_input, train_target)
print('1차 훈련 score : ', lr.score(train_input, train_target))
print('1차 테스트 score : ', lr.score(test_input, test_target))
print('길이 50인 물고기 무게 예측 :', *lr.predict([[50]]))
# lr.coef_ : 방정식 계수 / lr.intercept : y 절편 / 둘 다 ML 알고리즘이 찾은 값 -> 모델 파라미터(대부분의 ML 알고리즘은 최적의 모델 파라미터를 찾는 알고리즘)
print('계수 ,절편 :', lr.coef_, lr.intercept_)  # [39.01714496] -709.0186449535474
print()

# -------------------------- 선형 회귀 모델 그래프 -----------------------------------------------
r = [15, 50]
plt.scatter(train_input, train_target)
plt.scatter(50, 1241, marker='D')
plt.plot(r, lr.coef_*r + lr.intercept_, 'r')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# --------------------------- 2차 방정식 선형 회귀 모델 ------------------------------------------
# 2차 방정식 이므로 훈련세트를 제곱해서 1열 늘려 줘야 한다.
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
lr.fit(train_poly, train_target)
print('2차 훈련 score : ', lr.score(train_poly, train_target))
print('2차 테스트 score :', lr.score(test_poly, test_target))
print('길이 50인 물고기 무게 예측 :', *lr.predict([[50**2, 50]])) #1573
print('계수 ,절편 :', lr.coef_, lr.intercept_) #[  1.01433211 -21.55792498] 116.05021078278259

# ---------------------------2차 방정식 선형 회귀 모델 그래프 ---------------------------------------
r = np.arange(15, 50) # 리스트 사용시 연산 안됨
plt.scatter(train_input, train_target)
plt.scatter(50, 1573, marker='D')
plt.plot(r, lr.coef_[0]*(r**2) +lr.coef_[1]*r + lr.intercept_, 'r')
plt.show()
