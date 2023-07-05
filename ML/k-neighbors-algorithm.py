import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# 도미 길이 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
# 도미 무게 데이터
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
# 빙어 길이 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# 빙어 무게 데이터
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

#------------------------ k 최근접 이웃 알고리즘 ---------------------------------------
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
# k 최근접 알고리즘에서 데이터는 이중리스트 사용
fish_data = [[length[i], weight[i]] for i in range(len(length))]

# 정답 리스트, 찾을려는 생선(도미) - 1, 빙어 - 0
# 찾을려는 대상 -> 1, 나머지 -> 0
fish_target = [1]*35 + [0]*14

# k-최근접 이웃 알고리즘 : 어떤 데이터의 답을 구할때 주위 데이터를 보고 다수를 차지하는 것을 정답으로 사용, 주위 데이터로 현재 데이터 판단
# 가장 가까운 직선거리의 데이터를 탐색하기 때문에 데이터 수가 너무 많으면 속도 느려짐
# 가까운 직선거리 데이터 설정(참고데이터를 40개로 설정, 디폴트는 5) -> kn = KNeighborsClassifier(n_neighbors=30)
kn = KNeighborsClassifier()
kn40 = KNeighborsClassifier(n_neighbors=40)

# fit() -> 주어진 데이터로 알고리즘(도미를 찾기 위한 기준)을 훈련
kn.fit(fish_data, fish_target)
kn40.fit(fish_data, fish_target)

# score() -> 모델 평가 메서드 : 0 ~ 1 사이의 값(정확도), 1 일 경우, 모든 데이터가 정확히 맞다 / 정확도 = (맞춘 데이터 개수) / (전체 데이터 개수)
print('kn.score : ', kn.score(fish_data, fish_target))
print(kn._fit_X)

print('kn40.score : ', kn40.score(fish_data, fish_target)) # 도미의 데이터 개수가 35개기 때문에 어떤 데이터를 넣어도 도미로 판단함(=빙어도 도미로 판단)

#predict() -> 주위 데이터를 보고 새로운 데이터의 정답을 예측
print(kn.predict([[30, 600]]))

#--------------- p64 실습 ------------------------------
"""
for i in range(5, 50):
    kn.n_neighbors = i
    a = kn.score(fish_data, fish_target)
    if a < 1:
        # i 번째 부터 정확도가 1보다 작아지기 시작함
        print(i)
        break
"""

plt.scatter(length, weight, c='skyblue') # 그래프 생성
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
