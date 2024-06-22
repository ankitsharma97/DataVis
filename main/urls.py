from django.urls import path
from . import views

urlpatterns = [
    path('upload', views.upload_file, name='upload_file'),
    path('display-plot/', views.display_plot, name='display_plot'),
]
