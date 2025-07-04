from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt
from data import get_data_generators
import collections
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32

# Load data
train_data, val_data = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Class indices:", train_data.class_indices)
print("Training class distribution:", collections.Counter(train_data.classes))

# Compute class weights based on training data
class_indices = train_data.classes  # This is correct now
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_indices),
    y=class_indices
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Load base model
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pretrained layers

# Add custom head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model using class weights
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight_dict  # ← This is the key line
)

# Evaluate performance
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

model.save('mask_model.h5')
