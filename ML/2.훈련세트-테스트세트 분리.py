import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
arr = [[1, 2], [4, 5], [7, 8], [10, 11]]
arr1 = np.array(arr)
print(arr1.shape)
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]
print(train_input)
print(train_target)

# fit() 은 train
kn.fit(train_input, train_target)
# score 은 test
print(kn.score(test_input, test_target)) # score : 0 / 왜냐하면 빙어의 길이와 크기를 훈련 set에 포함 시키지 않았기 때문
# 올바른 훈련 세트를 만들기 위해서는 골고루 섞여 있어야 한다 / 그렇지 않을 경우, 샘플링이 한쪽에 치우쳤다 -> 샘플링 편향

# 샘플링 편향을 피하기 위해 numpy 라이브러리 사용
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
#print(input_arr.shape)

# 렌덤시드 42로 지정
np.random.seed(42)
# 범위 0~48
index = np.arange(49)
# np.random.shuffle() -> 섞기
np.random.shuffle(index)

# 훈련 세트 만들기
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
# 테스트 세트 만들기
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
print(kn.predict(test_input))
print(test_target)


# 2차원 배열은 행과 열 인덱스를 콤마로 나누어 지정함
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

