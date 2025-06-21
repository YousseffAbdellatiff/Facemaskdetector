#from django.conf.urls import url
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views



urlpatterns = [
    path('mainPage', views.mainPage, name='mainPage')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)