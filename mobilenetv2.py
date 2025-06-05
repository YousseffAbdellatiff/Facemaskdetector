from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from data import get_data_generators

IMG_SIZE = 224
BATCH_SIZE = 32

train_data, val_data = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load MobileNetV2 base model with pre-trained ImageNet weights
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model

# Add custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

loss, accuracy = model.evaluate(val_data)
print('Validation Accuracy =', accuracy)

# Plot loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()