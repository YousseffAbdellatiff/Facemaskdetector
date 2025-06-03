from PIL import Image
import os

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

# Clean both class folders
remove_non_images("dataset/with_mask")
remove_non_images("dataset/without_mask")

# ✅ Add this missing line
data_path = "dataset"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Preview sample
images, labels = next(train_data)
plt.imshow(images[0])
plt.title(f"Label: {int(labels[0])}")
plt.axis("off")
plt.show()
