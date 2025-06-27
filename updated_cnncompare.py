
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from data import get_data_generators
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# Model builder with optional deeper structure
def build_model(num_dense_layers=2, dropout_rate=0.5):
    layers = [
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(dropout_rate),
    ]
    if num_dense_layers == 3:
        layers.append(keras.layers.Dense(32, activation='relu'))
        layers.append(keras.layers.Dropout(dropout_rate))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))

    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load data
train_data, val_data = get_data_generators(data_path="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Train model
model = build_model(num_dense_layers=3)
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Evaluate model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Prediction for evaluation
val_data.reset()
predictions = model.predict(val_data)
y_pred = (predictions > 0.5).astype(int)
y_true = val_data.classes

# Classification report
print(classification_report(y_true, y_pred, target_names=['With Mask', 'Without Mask']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['With Mask', 'Without Mask'],
            yticklabels=['With Mask', 'Without Mask'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
