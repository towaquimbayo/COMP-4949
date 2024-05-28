# pages/urls.py
from django.urls import path
from .views import homePageView, aboutPageView, towaPageView, homePost, results

urlpatterns = [
    path("", homePageView, name="home"),
    path("about/", aboutPageView, name="about"),
    path("towa/", towaPageView, name="towa"),
    path("homePost/", homePost, name="homePost"),
    path("results/<int:choice>/<str:gmat>/", results, name="results"),
]
