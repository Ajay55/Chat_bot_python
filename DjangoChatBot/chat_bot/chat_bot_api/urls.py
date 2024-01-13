from django.urls import path
from .views import get_answer

urlpatterns = [
    path('get_answer/', get_answer, name='get_answer'),
]
