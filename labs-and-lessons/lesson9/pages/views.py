from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
import pickle
import pandas as pd


def homePost(request):
    # Create variable to store choice that is recognized through entire function.
    choice = -999
    gmat = -999  # Initialize gmat variable.

    try:
        # Extract value from request object by control name.
        currentChoice = request.POST["choice"]
        gmatStr = request.POST["gmat"]

        # Crude debugging effort.
        print("*** Years work experience: " + str(currentChoice))
        choice = int(currentChoice)
        gmat = float(gmatStr)

    # Enters 'except' block if integer cannot be created.
    except:
        return render(
            request,
            "home.html",
            {
                "errorMessage": "*** The choice was missing please try again",
                "mynumbers": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ],
            },
        )
    else:
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(
            reverse(
                "results",
                kwargs={"choice": choice, "gmat": gmat},
            )
        )


# Production version of results() function.
def results(request, choice, gmat):
    # print("*** Inside results()")
    # return render(request, "results.html", {"choice": choice, "gmat": gmat})

    print("*** Inside results()")
    # load saved model
    with open("../lesson9_model_pkl", "rb") as f:
        loadedModel = pickle.load(f)

    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=["gmat", "work_experience"])

    workExperience = float(choice)
    print("*** GMAT Score: " + str(gmat))
    print("*** Years experience: " + str(workExperience))
    singleSampleDf = singleSampleDf._append(
        {"gmat": gmat, "work_experience": workExperience}, ignore_index=True
    )

    singlePrediction = loadedModel.predict(singleSampleDf)

    print("Single prediction: " + str(singlePrediction))

    return render(
        request,
        "results.html",
        {"choice": workExperience, "gmat": gmat, "prediction": singlePrediction},
    )


def homePageView(request):
    return render(
        request,
        "home.html",
        {
            "mynumbers": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "firstName": "Towa",
            "lastName": "Quimbayo",
        },
    )


def aboutPageView(request):
    # return request object and specify page.
    return render(request, "about.html")


def towaPageView(request):
    return render(request, "towa.html")
