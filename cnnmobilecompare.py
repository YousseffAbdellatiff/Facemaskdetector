from tensorflow import keras
from tensorflow.keras import layers
from data import get_data_generators
from matplotlib import pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# Get new generators for each model to avoid exhaustion
train_data_cnn, val_data_cnn = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
train_data_mnv2, val_data_mnv2 = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- Custom CNN (3 dense layers) ---
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_cnn = cnn_model.fit(
    train_data_cnn,
    validation_data=val_data_cnn,
    epochs=EPOCHS,
    verbose=1
)

# --- MobileNetV2 ---
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

mobilenet_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
mobilenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_mnv2 = mobilenet_model.fit(
    train_data_mnv2,
    validation_data=val_data_mnv2,
    epochs=EPOCHS,
    verbose=1
)

# --- Plot accuracy comparison ---
plt.plot(history_cnn.history['val_accuracy'], label='CNN 3-layer val_acc')
plt.plot(history_mnv2.history['val_accuracy'], label='MobileNetV2 val_acc')
plt.plot(history_cnn.history['accuracy'], '--', label='CNN 3-layer train_acc')
plt.plot(history_mnv2.history['accuracy'], '--', label='MobileNetV2 train_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN (3-layer) vs MobileNetV2 Accuracy')
plt.legend()
plt.show()