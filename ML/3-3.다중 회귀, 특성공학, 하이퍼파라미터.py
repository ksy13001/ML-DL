import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
"""
선형 회귀 알고리즘에서 더 많은 특성을 훈련 시키면 더 강력한 성능을 낸다. 하지만 특성이 너무 많으면 규제가 필요하다.
규제를 위해서는 릿지 회귀, 라쏘 회귀를 사용한다.
릿지 : 계수를 제곱한 값을 기준으로 규제
라쏘 : 계수의 절댓값을 기준으로 규제 (계수를 0으로 만들수도 있음)
하이퍼 파라미터 : 머신러닝 알고리즘이 학습하지 않는 파라미터, 사람이 사전에 지정해야함(ex: 릿지, 라쏘의 규제 강도, alpha)
"""

# ------------------------------------- 판다스를 사용하여 데이터 세트 준비 --------------------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/perch_full.csv')
# 넘파이로 변환
perch_full = df.to_numpy()

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
# --------------------------------- 변환기(특성을 만들 거나 전처리)----------------------------------------------------------
# PolynomialFeatures -> 각 특성을 제곱, 서로 곱한 항 추가
poly = PolynomialFeatures(degree=5, include_bias=False)  # include_bias = False -> 절편 x
# 훈련을 해야 변환이 가능하다 / fit_transform 으로 대체 가능함
train_poly = poly.fit_transform(train_input)
test_poly = poly.fit_transform(test_input)
print(train_input.shape)
print(train_poly.shape)
# 특성이 어떻게 조합되었는지 보기
#print(poly.get_feature_names_out())
# ---------------------------- 다중 회귀 모델 훈련-------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# -144 / 특성의 개수를 너무 과도하게 늘리면 훈련세트에 너무 과대 적합이므로 음수가 나옴 -> 규제 필요
# ------------------------------ 모델 규제(regularization)----------------------------------------------------------------
# 규제를 할때 특성의 스케일을 정규화 해야 함 / 특성을 표준점수로 바꿔도 되지만 StandardScaler 클래스 사용
# 선형 회귀 모델에 규제를 추가한 모델 -> 릿지(ridge), 라쏘(lasso) / 릿지 -> 계수를 제곱한 값 기준 규제 / 라쏘 -> 계수의 절댓값 기준 규제
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
# 릿지, 라쏘는 alpha 변수로 규제 강도 조정
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))    # 음수 x
# --------------------------------------------------- 리짓 회귀 ----------------------------------------------------------
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for a in alpha_list:
    ridge = Ridge(alpha=a)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
# x축 alpha 값이 0.001, 0.01 식이면 그래프 그리기 힘드므로 자연로그 지수로 표현 / 0.001 -> -3, 0.01 -> -2, 0.1 -> -1
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
# 그래프 결과 -> alpha = 0.1(-1) 일때, 두 그래프가 가장 가까움(= 가장 차이 적음) -> alpha 값 0.1 로 설정
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print('ridge(alpha = 0.1) train score: ', ridge.score(train_scaled, train_target))
print('ridge(alpha = 0.1) test  score: ', ridge.score(test_scaled, test_target))

# ------------------------------------------ 리쏘 회귀 -------------------------------------------------------------------
train_score = []
test_score = []
for a in alpha_list:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
# 그래프 결과 -> alpha = 10(1) 일때, 두 그래프가 가장 가까움(= 가장 차이 적음) -> alpha 값 10 으로 설정
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print('lasso(alpha = 10) train score: ', lasso.score(train_scaled, train_target))
print('lasso(alpha = 10) test  score: ', lasso.score(test_scaled, test_target))

