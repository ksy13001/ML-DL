import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# ----------------------------------------------- 데이터 전처리 ----------------------------------------------------------
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish)
fish_input = fish[['Weight', 'Length', 'Diagonal','Height','Width' ]].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
# 표준화 전처리시 항상 훈련 세트로 통계값 사용
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# ------------------------------------- 확률적 경사 하강법 분류 클래스 : SGDClassifier --------------------------------------
# from sklearn.linear_model import SGDClassifier
# SGDClassifier -> 2개의 매개변수 있음 / loss -> 손실 함수 종류 지정 / max_iter -> 수행할 에포크(훈련세트를 모두 사용) 횟수 지정(= 훈련세트 반복횟수)
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)

sc.fit(train_scaled, train_target)
# 훈련 세트 점수 : 0.773109243697479
print(sc.score(train_scaled, train_target))
# 테스트 세트 점수 : 0.775
print(sc.score(test_scaled, test_target))
# 훈련, 테스트 세트 점수 부족함 -> partial_fit() : 모델을 이어서 훈련할 때 사용(확률적 경사 하강법은 점진적 학습이 가능함), 에포크 + 1
sc.partial_fit(train_scaled, train_target)
# 훈련 세트 점수 : 0.8151260504201681
print('epoch 10회 훈련 세트 정확도: ',sc.score(train_scaled, train_target))
# 테스트 세트 점수 : 0.85
print('epoch 10회 테스트 세트 정확도 :', sc.score(test_scaled, test_target))
"""
    에포크 횟수 너무 적음 -> 훈련 세트를 너무 적게 학습 -> 과소 적합
    에포크 횟수 너무 많음 -> 훈련 세트를 너무 많이 학습 -> 과대 적합
    -> 과대 적합이 시작되기 전에 훈련을 멈춰야 함 ->  조기 종료
"""

# ------------------------------------------ 에포크 횟수에 따른 정확도 차이 -------------------------------------------------
sc = SGDClassifier(loss='log_loss', random_state=42)
# 훈련 세트 점수 리스트
train_score = []
# 테스트 세트 점수 리스트
test_score = []
# np.unique(a) -> a 에서 고유값만 모아서 1차원 shape 로 변환 : train_target 에서 고유값(Bream, Smelt ...) 만 모아서 7개 생선 리스트 만듬
classes = np.unique(train_target)

for i in range(300):
    # 에포크 10 + 0 ~ 300 번 수행시 훈련, 테스트 score 리스트에 추가
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# --------------------------------------------  그래프 작성 --------------------------------------------------------------
# 그래프 초반에는 적은 에포크 횟수로 인해 과소 적합이 일어남, 에포크 횟수 100회 이후에는 과대 적합이 일어남 -> 반복횟수 100에 맞추고 모델 다시 훈련
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('score')
plt.show()
# -----------------------------------------에포크 100회 반복시 정확도-------------------------------------------------------
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)    # tol : 성능이 향상될 최솟값
sc.fit(train_scaled, train_target)
print('epoch 100회 훈련 세트 정확도: ', sc.score(train_scaled, train_target))
print('epoch 100회 테스트 세트 정확도: ', sc.score(test_scaled, test_target))

# ---------------------------------------- loss = 'hinge' 사용(힌지 손실) ------------------------------------------------
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print('힌지 손실 훈련 세트 정확도 :', sc.score(train_scaled, train_target))
print('힌지 손실 테스트 세트 정확도 :', sc.score(test_scaled, test_target))
