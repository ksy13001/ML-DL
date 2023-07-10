import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
print(np.column_stack([[1, 4, 7], [2, 5, 8], [3, 6, 9]]))
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
# np.column_stack() -> 튜플로 연결된 리스트들을 일렬로 세우고 차례대로 연결 (각 리스트에서 하나씩 뽑아내서 만듬)
fish_data = np.column_stack((fish_length, fish_weight))
# np.concatenate() -> 입력받은 리스트 뒤에 리스트를 차례대로 붙임
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# train_test_split() -> 전달받은 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 나눔, stratify = 타겟 데이터 -> 타겟 데이터를 샘플 비율에 맞게 조정함
# stratify -> 훈련 데이터가 적거나 샘플 개수가 적을때 사용
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42, stratify=fish_target)
kn = KNeighborsClassifier() # n_neighbor 기본값 = 5
# x 축 범위 설정(y축과 같게) / x축은 범위가 작은데 비해(10 ~ 40), y축은 범위가 큼(0~1000) -> 거리를 제대로 잴수가 없음 (= 스케일이 다르다)
# -> 거리 기반의 알고리즘들은 샘플 간의 거리에 영향을 많이 받으므로 특성값을 일정한 기준으로 맞춰줘야함 -> 데이터 전처리

# 평균 계산 / axis = 0 -> 각 열의 평균, axis = 1 -> 각 행의 평균
mean = np.mean(train_input, axis=0)
# 표준 편차 계산
# 평균 과 표준편차 모두 훈련 세트에서 계산
std = np.std(train_input, axis=0)
# 훈련 세트 표준 점수 계산 / numpy 는 연산시 모든 데이터에 적용 -> 브로드캐스팅
train_scaled = (train_input - mean) / std
# 표준 점수로 변환한 train 데이터로 모델 훈련
kn.fit(train_scaled, train_target)
# 테스트 세트도 표준점수로 변환해야함
test_scaled = (test_input - mean) / std
# 테스트 샘플 변환
new = (([25, 150] - mean)/ std)

print('score : ', kn.score(test_scaled, test_target))
print('predict : ', kn.predict([new]))
distances, indexes = kn.kneighbors([new])
# ---------------------그래프-----------------------
plt.scatter(train_scaled[:,0], train_scaled[:,1]) # train_input 에서 인덱스가 0, 1인 값들
plt.scatter(new[0], new[1], marker='*')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
