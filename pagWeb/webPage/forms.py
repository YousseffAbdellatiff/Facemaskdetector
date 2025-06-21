from django import forms
from .models import UploadedImage

class UploadImageForms(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']
