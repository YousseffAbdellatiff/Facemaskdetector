from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

#This line loads the created model
model = load_model("mask_model.h5")

img_path = 'testImage.jpg'

#For error handling in case the path to the file is not correct
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
else:
    img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize

prediction = model.predict(img_array)

confidence = prediction[0][0]
if confidence > 0.5:
    print(f"Prediction: No Mask ({confidence:.2f} confidence)")
else:
    print(f"Prediction: With Mask ({1 - confidence:.2f} confidence)")

if prediction[0][0] > 0.5:
    print("Prediction: No mask")

else:
    print("Prediction: With Mask")