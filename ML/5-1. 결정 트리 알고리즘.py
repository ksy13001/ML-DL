import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 결정 트리 알고리즘 ->>> 데이터 표준화 전처리 필요 X
# 불순도를 기준으로 샘플을 나눔, 불순도는 클래스의 비율로 계산, 클래스 비율에 따라 달라지므로 스케일 필요 x
# ---------------------------------------------- 데이터 세트 준비, 전처리 --------------------------------------------------
wine = pd.read_csv('http://bit.ly/wine_csv_data')
# info() -> 데이터 프레임 각 열과 누락된 데이터 확인
print(wine.info())
# describe() -> 열에 대해 간단한 통계(최소, 최대, 평균값)
print(wine.describe())
wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
wine_target = wine['class'].to_numpy()
# test_size = 0.2 -> 20% 정도를 테스트 세트로 나눔
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, test_size=0.2, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# print(train_input.shape, test_input.shape) -> (5197, 3) (1300, 3) -> 훈련세트: 5197개, 테스트세트: 1300개

# ------------------------------------------------- 로지스틱 회귀 모델 생성 ------------------------------------------------
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print('로지스틱 훈련 세트 score: ', lr.score(train_scaled, train_target))
print('로지스틱 테스트 세트 score:', lr.score(test_scaled, test_target))

# ------------------------------------------------ 결정 트리 모델 생성 ----------------------------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print('결정트리 훈련 세트 score: ', dt.score(train_scaled, train_target))
print('결정트리 테스트 세트 score: ', dt.score(test_scaled, test_target))

# ------------------------------------------------ 결정 트리 그래프 생성 --------------------------------------------------
# 이미지 크기 설정
plt.figure(figsize=(10, 7))
# max_depth=i -> 루트 노드를 i개 층 확장, filled=True-> 양성 클래스 비율이 높을수록 색 진하게, feature_name -> 특성 이름 전달
# 리프 노드에서 가장 많은 클래스가 예측 클래스(양성클래스가 가장 많으면 양성클래스가 예측 클래스)
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
"""
*노드 구성
-테스트 조건(sugar)
-불순도(gini) -> 1-(음성 클래스 비율**2 + 양성 클래스 비율**2) = 1-((1258/5197)**2 + (3939/5197)**2) = 0.367
-총 샘플 수(samples)
-클래스별 샘플 수(value=[1258, 3939]) 음성클래스(레드와인)=1258, 양성 클래스(화이트와인)=3939
"""
"""
*불순도
-불순도=0.5 -> 어떤 노드의 두 클래스 비율이 1/2 -> 최악
-불순도=0   -> 한 클래스가 모두 차지, 불순도 최소 -> 순수 노드

정보 이득 -> 부모와 자식 노드의 불순도 차이
결정 트리 알고리즘은 불순도 기준을 사용해 정보 이득이 최대가 되도록 노드를 분할, 노드를 순수하게 나눌수록 정보이득 커짐
새로운 샘플을 예측할 때 노드의 질문에 따라 트리 이동, 마지막에 도달한 노드의 클래스 비율을 보고 예측을 만듬
"""

# ------------------------------------------- 트리 가지치기 --------------------------------------------------------------
# 훈련세트 score > 테스트세트 score -> 일반화 x, 과대적합 -> 트리 가지치기 필요함 -> 트리의 최대 깊이 지정
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print('max_depth=3')
print('훈련 score:', dt.score(train_scaled, train_target))
print('테스트 score:', dt.score(test_scaled, test_target))
# 결정트리의 특성 중요도
print(dt.feature_importances_)  # [0.12345626 0.86862934 0.0079144 ]
# 루트 노드에서도 sugar 기준으로 분류 -> 가장 중요한 특성, 특성 중요도에서도 0.86 으로 가장 큰 수치

plt.figure(figsize=(15, 15))
plot_tree(dt,filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
