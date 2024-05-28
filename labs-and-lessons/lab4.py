# Exercise 1, 2, 3, 4
def ex1():
    import warnings

    warnings.filterwarnings("ignore")

    from pandas import read_csv
    import matplotlib.pyplot as plt
    import statsmodels.tsa.arima.model as sma
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    import numpy as np
    import pmdarima as pm

    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    df = read_csv(PATH + "daily-total-female-births.csv", header=0, index_col=0)
    print(df.head())
    print(df.describe())

    # Split the data set so the test set is 7.
    TEST_DAYS = 7

    X_train = df[0 : len(df) - TEST_DAYS]
    y_train = df[0 : len(df) - TEST_DAYS]
    X_test = df[len(df) - TEST_DAYS :]
    y_test = df[len(df) - TEST_DAYS :]

    # Create a list with the training array.
    predictions = []

    for i in range(len(X_test)):
        print("\n*****************************************")
        print("Iteration: " + str(i))
        print("Length of training data: " + str(len(X_train)))
        model = pm.auto_arima(
            X_train,
            start_p=1,
            start_q=1,
            test="adf",
            max_p=3,
            max_q=3,
            m=0,
            start_P=0,
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        print(model.summary())

        yhat = model.predict(start=len(X_train), end=len(X_train))

        if i < len(X_test):
            test_row = X_test.iloc[i]
            X_train = X_train._append(test_row, ignore_index=True)
            predictions.append(yhat.iloc[0])
        else:
            break

    #################################################################

    rmse = sqrt(mean_squared_error(X_test, predictions))
    print("Test RMSE: %.3f" % rmse)

    plt.plot(X_test, label="Actual", marker="o", color="blue")
    plt.plot(predictions, label="Predictions", marker="o", color="orange")
    plt.legend()
    plt.title("AR Model")
    plt.show()


def ex5():
    import pandas as pd

    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE_NAME = "Energy_Production.csv"

    # Import
    data = pd.read_csv(PATH + FILE_NAME, index_col=0)
    data.index = pd.to_datetime(data.index)
    print(data.head())
    print(data.describe())


def ex6():
    import pandas as pd
    import matplotlib.pyplot as plt

    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE_NAME = "Energy_Production.csv"

    # Import
    data = pd.read_csv(PATH + FILE_NAME, index_col=0)
    data.index = pd.to_datetime(data.index)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)

    # Usual Differencing
    axes[0].plot(data[:], label="Original Series")
    axes[0].plot(data[:].diff(1), label="Usual Differencing")
    axes[0].set_title("Usual Differencing")
    axes[0].legend(loc="upper left", fontsize=10)

    # Seasonal Differencing
    axes[1].plot(data[:], label="Original Series")
    axes[1].plot(data[:].diff(12), label="Seasonal Differencing", color="green")
    axes[1].set_title("Seasonal Differencing")
    plt.legend(loc="upper left", fontsize=10)
    plt.show()


def ex7():
    import warnings

    warnings.filterwarnings("ignore")

    import pandas as pd

    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE_NAME = "Energy_Production.csv"
    import matplotlib.pyplot as plt

    # Import
    data = pd.read_csv(PATH + FILE_NAME, index_col=0)
    data.index = pd.to_datetime(data.index)

    import pmdarima as pm

    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(
        data,
        start_p=1,
        start_q=1,
        test="adf",
        max_p=3,
        max_q=3,
        m=12,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    print(smodel.summary())


def ex8():
    import warnings

    warnings.filterwarnings("ignore")

    import pandas as pd

    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE_NAME = "Energy_Production.csv"
    import matplotlib.pyplot as plt

    # Import
    data = pd.read_csv(PATH + FILE_NAME, index_col=0)
    data.index = pd.to_datetime(data.index)

    import pmdarima as pm

    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(
        data,
        start_p=1,
        start_q=1,
        test="adf",
        max_p=3,
        max_q=3,
        m=12,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    # Forecast
    NUM_TIMESTEPS = 24
    fitted, confint = smodel.predict(n_periods=NUM_TIMESTEPS, return_conf_int=True)
    index_of_fc = pd.date_range(data.index[-1], periods=NUM_TIMESTEPS, freq="MS")

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(data)
    plt.plot(fitted_series, color="darkgreen")
    plt.fill_between(
        lower_series.index, lower_series, upper_series, color="k", alpha=0.15
    )

    plt.title("SARIMA - Final Forecast of Energy Production")
    plt.show()


def main():
    # ex1()
    # ex5()
    # ex6()
    # ex7()
    ex8()


if __name__ == "__main__":
    main()
