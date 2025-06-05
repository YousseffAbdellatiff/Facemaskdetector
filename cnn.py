try:
    import tensorflow as tf
    from tensorflow import keras
    print("TensorFlow is already installed with version:", tf.__version__)
except ImportError:
    print("TensorFlow not found. Installing TensorFlow...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    
    # Import after installation
    import tensorflow as tf
    from tensorflow import keras
    print("TensorFlow has been installed with version:", tf.__version__)

from matplotlib import pyplot as plt
from data import get_data_generators

IMG_SIZE = 224
BATCH_SIZE = 32

train_data, val_data = get_data_generators(
    data_path="dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Number of output classes (binary classification: with mask and with no mask)
num_of_classes = 2

# Initialize a Sequential model (stack of layers)
model = keras.Sequential([

    # --- First Convolutional Block ---
    # Convolutional layer with 32 filters of size 3x3
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    # MaxPooling reduces the spatial dimensions by taking the max value over a 2x2 area
    keras.layers.MaxPooling2D(2, 2),

    # --- Second Convolutional Block ---
    # Convolutional layer with 64 filters
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Another pooling layer to reduce size and extract higher-level features
    keras.layers.MaxPooling2D(2, 2),

    # --- Flattening ---
    # Flatten the 2D feature maps into a 1D vector to feed into dense layers
    keras.layers.Flatten(),

    # --- Fully Connected Layer 1 ---
    # Dense (fully connected) layer with 128 units and ReLU activation
    keras.layers.Dense(128, activation='relu'),
    
    # Dropout randomly sets 50% of neurons to 0 during training to reduce overfitting
    keras.layers.Dropout(0.5),
    
    # --- Fully Connected Layer 2 ---
    # Another dense layer with 64 units
    keras.layers.Dense(64, activation='relu'),
    
    # Another dropout for regularization
    keras.layers.Dropout(0.5),

    # --- Third Fully Connected Layer ---
    keras.layers.Dense(32, activation='relu'),
    
    # Another dropout for regularization
    keras.layers.Dropout(0.5),


    # --- Output Layer ---
    # Final dense layer with 1 output unit and sigmoid activation
    # Sigmoid outputs a value between 0 and 1, suitable for binary classification
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# compile the neural network
model.compile(optimizer='adam',
              #loss='binary_crossentropy',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the neural network
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