from django.urls import path
from .views import home_page_view, home_page_post, results_page_view

urlpatterns = [
    path("", home_page_view, name="home"),
    path("homePost/", home_page_post, name="homePost"),
    path(
        "results/<int:department>/<int:previous_year_rating>/<int:awards_won>/<str:avg_training_score>/",
        results_page_view,
        name="results",
    ),
]
