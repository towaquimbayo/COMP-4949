from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from pickle import load
import pandas as pd
from keras.models import load_model


def home_page_view(request):
    return render(
        request,
        "home.html",
        {
            "department": [
                "Analytics",
                "Finance",
                "HR",
                "Operations",
                "Procurement",
                "Technology",
                "R&D",
                "Sales & Marketing",
            ],
            "previous_year_rating": [0, 1, 2, 3, 4, 5],
            "awards_won": [1, 0],
        },
    )


def home_page_post(request):
    departments = [
        "Analytics",
        "Finance",
        "HR",
        "Operations",
        "Procurement",
        "Technology",
        "R&D",
        "Sales & Marketing",
    ]
    try:
        department = int(departments.index(request.POST["department"]))
        awards_won = int(request.POST["awards_won"])
        previous_year_rating = int(request.POST["previous_year_rating"])
        avg_training_score = float(request.POST["avg_training_score"])
    except:
        return render(
            request,
            "home.html",
            {
                "errorMessage": "Please fill in all the fields.",
                "department": departments,
                "awards_won": [1, 0],
                "previous_year_rating": [0, 1, 2, 3, 4, 5],
            },
        )
    else:
        return HttpResponseRedirect(
            reverse(
                "results",
                kwargs={
                    "department": department,
                    "awards_won": awards_won,
                    "previous_year_rating": previous_year_rating,
                    "avg_training_score": avg_training_score,
                },
            )
        )


def results_page_view(
    request, department, awards_won, previous_year_rating, avg_training_score
):
    try:
        model = load_model("./model/best_model.h5")
        with open("./model/scaler.pkl", "rb") as file:
            scaler = load(file)

        df = pd.DataFrame(
            columns=[
                "department",
                "previous_year_rating",
                "awards_won?",
                "avg_training_score",
            ]
        )
        df = df._append(
            {
                "department": department,
                "previous_year_rating": previous_year_rating,
                "awards_won?": awards_won,
                "avg_training_score": avg_training_score,
            },
            ignore_index=True,
        )

        x = df.values
        x = scaler.transform(x)
        predictions = model.predict(x)

        print("Predictions:")
        for prediction in predictions:
            print(prediction)

        is_promoted = [1 if prediction > 0.5 else 0 for prediction in predictions]
        print("Is Promoted:")
        for promotion in is_promoted:
            print(promotion)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise Http404("Model not found")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise Http404("An error occurred")

    departments = [
        "Analytics",
        "Finance",
        "HR",
        "Operations",
        "Procurement",
        "Technology",
        "R&D",
        "Sales & Marketing",
    ]

    return render(
        request,
        "results.html",
        {
            "department": departments[department],
            "previous_year_rating": previous_year_rating,
            "awards_won": awards_won,
            "avg_training_score": avg_training_score,
            "is_promoted": is_promoted[0],
        },
    )
