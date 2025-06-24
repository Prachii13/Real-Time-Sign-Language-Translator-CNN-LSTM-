import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
import numpy as np
import os

# Dummy dataset generator
def load_data():
    X = np.random.rand(100, 30, 64, 64, 3)  # 100 videos, 30 frames each, 64x64 size
    y = np.random.randint(0, 2, 100)
    return X, tf.keras.utils.to_categorical(y, 2)

X, y = load_data()

model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(30, 64, 64, 3)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, validation_split=0.2)
model.save("model.h5")
print("✅ Model trained and saved.")
