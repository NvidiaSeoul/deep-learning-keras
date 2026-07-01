import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

(train_x, _ ),(test_x, _ ) = \
    fashion_mnist.load_data() # (28,28) 패션 관련 흑백 이미지

print(len(train_x)) # 60000  ==> ( 60000, 28, 28 )

train_x = train_x.reshape(-1,28,28,1) # -1 : 자동으로 결정해라
print(train_x.shape)

test_x = test_x.reshape(-1, 28, 28, 1)
print(test_x.shape)