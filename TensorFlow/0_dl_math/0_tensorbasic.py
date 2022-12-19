import numpy as np
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images 배열의 ndim 속성으로 축의 갯수 확인
print(train_images.ndim)

# train_images 배열의 크기
print(train_images.shape)

# 속성으로 데이터 타입 확인
print(train_images.dtype)

# Slicing, 11번째에서 101번째까지(101번은 포함x) 선택해 (90, 28, 28) 크기의 배열을 제작
my_slice = train_images[10:100]

# 이미지 오른쪽 아래 14x14 픽셀 선택
my_slice = train_images[:, 14:, 14:]

# 음수 인덱스 사용또한 가능, 현재 축의 끝에서 상대적 위치를 나타냄
# 정중앙에 위치한 14x14 픽셀 조각을 잘라냄
my_slice = train_images[:, 7:-7, 7:-7]
