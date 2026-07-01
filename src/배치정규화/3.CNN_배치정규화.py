from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation

(train_input, train_target), (test_input, test_target) = mnist.load_data()
print(train_input.shape)  # (60000, 28, 28 ) ==> 60000 개를 3000개로 줄여서 테스트
# import matplotlib.pyplot as plt
# plt.imshow(train_input[1], cmap='gray') # 손글씨 0 grayscale 데이터 ( 0 ~ 255값 )
# plt.show()
# # 훈련,테스트 데이터 shape 변경 및 정규화
train_input = train_input.reshape(-1, 28, 28, 1).astype('float32')[:3000] / 255
test_input = test_input.reshape(-1, 28, 28, 1).astype('float32')[:3000] / 255
print(train_input.shape)  # (60000, 28, 28 )
# 타깃 데이터 원핫 인코딩
train_target = to_categorical(train_target[:3000])
test_target = to_categorical(test_target[:3000])
print(train_target[1])

# Conv 신경망 설계
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_input, train_target, validation_data=(test_input, test_target), epochs=10,
          batch_size=10, verbose=1)

print('acc : ', model.evaluate(test_input, test_target)[1])

