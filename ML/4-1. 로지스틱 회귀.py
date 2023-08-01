import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, softmax

# 로지스틱 회귀 -> 선형 회귀 처럼 계산 한 값을 0 ~ 1 사이로 압축 시킴 -> 확률로 사용 가능
# 로지스틱 회귀 : 이진 분류 -> 하나의 선형 방정식 생성 -> 하나의 출력 값(z) - 시그모이드 함수에 적용 -> 0~1 사이 값 : 양성 클래스 확률


# ------------------------------------- 판다스를 사용하여 데이터 세트 준비 --------------------------------------------------
fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish.head())                처음 5개 행 출력
# print(pd.unique(fish['Species'])) Species 열에서 고유한 값 출력
# numpy 로 변환
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()
# 데이터 세트 생성
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
# StandardScaler 사용 -> 데이터 세트 표준화(전처리), 훈련 세트 통계 값으로 테스트 세트 변환
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# -------------------------------------- 최근접 이웃 분류기 ---------------------------------------------------------------
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print('훈련 세트 score:', kn.score(train_scaled, train_target))
print('테스트 세트 score:', kn.score(test_scaled, test_target))

# proba -> 클래스 별 확률값 반환 (테스트 세트 처음 5개 샘플 확률)
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
"""
i 샘플/ i클래스 확률 1번째클래스 
1번째 샘플          [[0.     0.     1.     0.     0.     0.     0.    ] 
2번째 샘플           [0.     0.     0.     0.     0.     1.     0.    ]
3번째 샘플           [0.     0.     0.     1.     0.     0.     0.    ]
4번째 샘플           [0.     0.     0.6667 0.     0.3333 0.     0.    ]
5번째 샘플           [0.     0.     0.6667 0.     0.3333 0.     0.    ]]

4번째 샘플이 3번째 클래스일 확률 -> 0.6667
"""
distances, indexes = kn.kneighbors(test_scaled[3:4])
# ----------------------------------- 로지스틱 회귀(시그모이드 함수)그래프 생성 -----------------------------------------------
# 로지스틱 회귀 그리기 : 확률을 나타내기 위해 사용,  z 가 아주 큰 음수 일때, 0 / 아주 큰 양수 일떄 1
# 출력 > 0.5 : 양성 클래스 / 출력 <= 0.5 : 음성 클래스
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
# plt.show()

# ------------------------------------ 불리언 인덱싱 ---------------------------------------------------------------------
"""
arr = np.array(['a', 'b', 'c', 'd', 'e'])
print(arr[[True, False, True, False, False]])   # [a, c]
"""
# 불리언 인덱싱으로 훈련 세트에서 도미, 빙어 행을 골라내기
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# -------------------------------------- 로지스틱 회귀 모델 훈련 -----------------------------------------------------------
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(train_bream_smelt[:5])
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
# 로지스틱 회귀 계수 확인
print(lr.coef_, lr.intercept_)
# [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
# 로지스틱 방정식 -> z = -0.404 * (Weight) - 0.576 * (Length) - 0.663 * (Diagonal) - 1.013 * (Height) - 0.732 * (Width) - 2.161

# z값 구하기 -> 이 z 값으로 시그모이드 함수에 통과 시키면 확률을 얻을 수 있음 : expit() / from scipy.special import expit
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
print(lr.classes_)  # ['Bream', 'Smelt']
# expit() -> 시그모어드 함수
print(expit(decisions))
print(lr.predict_proba(train_bream_smelt[:5]))

# --------------------------------------- 로지스틱 회귀 다중 분류 ----------------------------------------------------------
lr = LogisticRegression(C=20, max_iter=1000)  # C: 규제 매개변수 : 작을수록 규제가 커짐(릿지의 alpha와 반대), 기본값은 1
lr.fit(train_scaled, train_target)
# 로지스틱 훈련 세트 점수
print(lr.score(train_scaled, train_target))
# 테스트 세트 점수
print(lr.score(test_scaled, test_target))

print(test_scaled[:5])
print(lr.predict(test_scaled[:5]))
print(lr.classes_)
# lr 테스트 세트 예측 확률/ 소수점 4번째 자리에서 반올림
print(np.round(lr.predict_proba(test_scaled[:5]), decimals=3))
"""
로지스틱 회귀에서 테스트 세트의 처음 5개 샘플의 각 클래스별 예측 확률(7 개의 생선에 관한 확률)
['Bream'  'Parkki'   'Perch'  'Pike'   'Roach'  'Smelt' 'Whitefish']
[[0.      0.014       0.841    0.       0.136    0.007    0.003]
 [0.      0.003       0.044    0.       0.007    0.946    0.   ]
 [0.      0.          0.034    0.935    0.015    0.016    0.   ]
 [0.011   0.034       0.306    0.007    0.567    0.       0.076]
 [0.      0.          0.904    0.002    0.089    0.002    0.001]]
"""
print(lr.coef_.shape, lr.intercept_.shape)

# z 값 구하기 / z 값 -> 곧 예측 확률
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 소프트 맥스 함수 적용/ from scipy.special import softmax / axis -> 소프트맥스를 계산할 축 설정, 지정 x -> 배열 전체 계산
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
