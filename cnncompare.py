from matplotlib import pyplot as plt
from tensorflow import keras
from data import get_data_generators

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

def build_model(num_dense_layers=2):
    layers = [
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
    ]
    if num_dense_layers == 3:
        layers.append(keras.layers.Dense(32, activation='relu'))
        layers.append(keras.layers.Dropout(0.5))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2-layer model (matches cnn.py: 128 + 64 dense layers)
train_data_2, val_data_2 = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
model_2 = build_model(num_dense_layers=2)
history_2 = model_2.fit(train_data_2, validation_data=val_data_2, epochs=EPOCHS, verbose=1)

# 3-layer model (adds Dense(32) + Dropout)
train_data_3, val_data_3 = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
model_3 = build_model(num_dense_layers=3)
history_3 = model_3.fit(train_data_3, validation_data=val_data_3, epochs=EPOCHS, verbose=1)

# Plot accuracy comparison
plt.plot(history_2.history['val_accuracy'], label='2-layer val_acc')
plt.plot(history_3.history['val_accuracy'], label='3-layer val_acc')
plt.plot(history_2.history['accuracy'], '--', label='2-layer train_acc')
plt.plot(history_3.history['accuracy'], '--', label='3-layer train_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('2-layer vs 3-layer CNN Accuracy')
plt.legend()
plt.show()