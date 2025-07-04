from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

#This line loads the created model
model = load_model("mask_model.h5")

img_folder = 'testImages'

class_names = ['Incorrect', 'Masked', 'Without Mask']

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
                predict_index = np.argmax(prediction[0])
                confidence = float(prediction[0][predict_index])
                label = class_names[predict_index]

                if confidence < 0.7:
                    label = "Unknown"
                else:
                    print(f"{filename}: {label} (Confidence: {confidence:.2f})")

                plt.imshow(img)
                plt.title(f"{filename}\n{label} ({confidence:.2f})")
                plt.axis('off')
                plt.show()

            except Exception as e:
                print(f"Error processing {filename}: {e}")




