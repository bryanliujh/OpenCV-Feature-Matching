from django.contrib import admin
from django.urls import path
from myApp import  views

urlpatterns = [
    path ('',views.testView),
    path('admin/', admin.site.urls),
]
