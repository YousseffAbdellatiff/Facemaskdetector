from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === STEP 1: Clean corrupt or non-image files ===
def remove_non_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for fname in files:
            full_path = os.path.join(root, fname)
            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception:
                print(f"❌ Removing invalid file: {full_path}")
                os.remove(full_path)

# === STEP 2: Clean all class folders ===
# Assuming structure: dataset/dataset/with_mask etc.
base_path = "dataset"  # adjust if different
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        remove_non_images(folder_path)

# === STEP 3: Setup Data Augmentation ===
IMG_SIZE = 224
BATCH_SIZE = 32
data_path = base_path  # e.g., dataset/dataset

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# === STEP 4: Load training and validation data (multi-class) ===
train_data = datagen.flow_from_directory(
    data_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # MULTI-CLASS mode
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === STEP 5: Preview sample image ===
images, labels = next(train_data)
plt.imshow(images[0])
plt.title(f"One-hot label: {labels[0]}")
plt.axis("off")
plt.show()
