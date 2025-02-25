import numpy as np
import pickle
import cv2, os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
NUM_CLASSES = 36  # A-Z + 0-9
IMAGE_SIZE = 50   # Ensure images are 50x50

# 🚀 Improved CNN Model
def cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), activation='relu'),
        BatchNormalization(),  # ✅ Normalization for stability
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        Flatten(),

        Dense(256, activation='relu'),  # ✅ Increased neurons
        Dropout(0.3),  # ✅ Increased dropout to prevent overfitting

        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0005),  # ✅ Lower learning rate for better convergence
                  metrics=['accuracy'])

    filepath = "cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    return model, [checkpoint]

# 🚀 Train Model
def train():
    # Load dataset
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f))

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f))

    # 🚀 Ensure correct shape
    print(f"✅ Final train images: {train_images.shape}")
    print(f"✅ Final train labels: {train_labels.shape}")
    print(f"✅ Final val images: {val_images.shape}")
    print(f"✅ Final val labels: {val_labels.shape}")

    # 🚀 Ensure images are correctly reshaped for CNN
    train_images = train_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0  # ✅ Normalize pixel values
    val_images = val_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0

    # Load Model
    model, callbacks_list = cnn_model()
    model.summary()

    # Train Model
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels),
              epochs=20, batch_size=128, callbacks=callbacks_list)  # ✅ More epochs, smaller batch size

    # Evaluate Model
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"📉 CNN Error: {100 - scores[1] * 100:.2f}%")

# 🚀 Run Training
train()
