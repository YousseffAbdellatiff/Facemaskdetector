from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import UploadImageForms
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("mask_model.h5")

def mainPage(request):
    label = None
    confidence = None
    image_url = None

    if request.method == 'POST':
        form = UploadImageForms(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            
            
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
            os.makedirs(upload_dir, exist_ok=True)

            fs = FileSystemStorage(location=upload_dir, base_url='/media/temp_uploads/')
            filename = fs.save(img_file.name, img_file)
            img_path = fs.path(filename)
            image_url = fs.url(filename) 

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            label = "Masked" if prediction[0][0] < 0.5 else "Not wearing mask"
            confidence = round(1 - prediction[0][0] if label == 'Masked' else prediction[0][0], 2)

    else:
        form = UploadImageForms()

    return render(request, 'webPage/mainPage.html', {
        'form': form,
        'label': label,
        'confidence': confidence,
        'image_url': image_url
    })
