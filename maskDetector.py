from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

#This line loads the created model
model = load_model("mask_model.h5")

img_folder = 'testImages'

#For error handling in case the path to the file is not correct
if not os.path.exists(img_folder):
    print(f"Image not found: {img_folder}")
    sys.exit(1)
else:
    for filename in os.listdir(img_folder):
        if filename.lower().endswith(('.jpg','.png','.jpeg')):
            img_path = os.path.join(img_folder, filename)
            try:
                img = image.load_img(img_path, target_size=(224,224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)/255.0
                
                prediction = model.predict(img_array)
                label = "Masked" if prediction[0][0] < 0.5 else "Not wearing mask"

                print(f"{filename}: {label} (Confidence: {1 - prediction[0][0] if label == 'Masked' else prediction[0][0]:.2f})")

                plt.imshow(img)
                plt.title(f"{filename}\n{label}")
                plt.axis('off')
                plt.show()

            except Exception as e:
                print(f"Error processing {filename}: {e}")




