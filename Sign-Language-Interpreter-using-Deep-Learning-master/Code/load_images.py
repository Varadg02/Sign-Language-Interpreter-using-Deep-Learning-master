import cv2
from glob import glob
import numpy as np
import pickle
import os
from sklearn.utils import shuffle
from keras.utils import np_utils

# Define label mapping for A-Z and 0-9
gesture_labels = {char: idx for idx, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")}
NUM_CLASSES = 36  # A-Z (26) + 0-9 (10)
IMAGE_SIZE = 50  # Ensure all images are 50x50

def pickle_images_labels():
    images_labels = []
    images = glob("gestures/*/*.jpg")
    images.sort()

    for image in images:
        folder_name = os.path.basename(os.path.dirname(image))
        if folder_name in gesture_labels:
            label = gesture_labels[folder_name]
        else:
            print(f"⚠️ Skipping unknown label: {folder_name}")
            continue
        
        img = cv2.imread(image, 0)  # Read in grayscale
        if img is None:
            print(f"⚠️ Skipping corrupted image: {image}")
            continue  # Skip unreadable images

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to 50x50
        images_labels.append((np.array(img, dtype=np.uint8), label))

    return images_labels

# Load and shuffle dataset
images_labels = pickle_images_labels()
images_labels = shuffle(images_labels)

# Split into images and labels
images, labels = zip(*images_labels)

# Ensure labels match images BEFORE encoding
assert len(images) == len(labels), f"❌ Mismatch: {len(images)} images, {len(labels)} labels"

# Convert to NumPy arrays
labels = np.array(labels)  
images = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)  # Reshape for CNN input

# ✅ Apply One-Hot Encoding ONLY ONCE
labels = np_utils.to_categorical(labels, num_classes=NUM_CLASSES)

print("✅ Final image count:", len(images))
print("✅ Final label count:", len(labels))
print("✅ Label shape after encoding:", labels.shape)  # Should print (Total images, 36)

# Split into training & validation sets
split_index = int(5/6 * len(images))  # 5:1 Train-Val Split
train_images, val_images = images[:split_index], images[split_index:]
train_labels, val_labels = labels[:split_index], labels[split_index:]

# Save datasets
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)

print(f"✅ Dataset successfully processed and saved! 🚀")
print(f"📊 Total images: {len(images)}")
print(f"📈 Training images: {len(train_images)}, Validation images: {len(val_images)}")
