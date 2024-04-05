from django.urls import include, path
from django.views.decorators.csrf import csrf_exempt
from .views import UploadParse

urlpatterns = [
        path('upload-file/',
            csrf_exempt(UploadParse.as_view()),
            name='upload-file'),
        ]
