from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os

def display_plot(request):
    image_path = os.path.join('images', 'boxplot.png')
    return render(request, 'display_plot.html', {'image_path': image_path})


def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_url = fs.url(filename)
        return render(request, 'upload_success.html', {'file_url': file_url})
    return render(request, 'index.html')
