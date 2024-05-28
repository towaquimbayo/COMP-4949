def ex1():
    import warnings
    import numpy as np
    from scipy import stats
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    import datetime
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until pandas_datareader is fixed.
    from sklearn.metrics import mean_squared_error

    warnings.filterwarnings("ignore")

    def getStock(stk, ttlDays):
        numDays = int(ttlDays)
        # Only gets up until day before during
        # trading hours
        dt = datetime.date.today()
        # For some reason, must add 1 day to get current stock prices
        # during trade hours. (Prices are about 15 min behind actual prices.)
        dtNow = dt + datetime.timedelta(days=1)
        dtNowStr = dtNow.strftime("%Y-%m-%d")
        dtPast = dt + datetime.timedelta(days=-numDays)
        dtPastStr = dtPast.strftime("%Y-%m-%d")
        yfin.pdr_override()
        df = pdr.get_data_yahoo(stk, start=dtPastStr, end=dtNowStr)
        return df

    def predictAndEvaluate(model, test, title, start, end):
        print("\n***" + title)
        print(model.summary())
        predictions = model.predict(start=start, end=end, dynamic=True)
        mse = mean_squared_error(predictions, test)

        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))
        return rmse, predictions

    def showPredictedAndActual(actual, predictions, ar, ma):
        indicies = list(actual.index)
        plt.title("AR: " + str(ar) + " MA: " + str(ma))
        plt.plot(indicies, predictions, label="predictions", marker="o")
        plt.plot(indicies, actual, label="actual", marker="o")
        plt.legend()
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.show()

    stkName = "MSFT"
    dfStock = getStock(stkName, 400)

    # Split the data.
    NUM_TEST_DAYS = 5
    lenData = len(dfStock)
    dfTrain = dfStock.iloc[0 : lenData - NUM_TEST_DAYS, :]
    dfTest = dfStock.iloc[lenData - NUM_TEST_DAYS :, :]

    modelStats = []
    for ar in range(0, 5):
        for ma in range(0, 5):
            model = ARIMA(dfTrain["Open"], order=(ar, 0, ma)).fit()
            title = str(ar) + "_0_" + str(ma)
            start = len(dfTrain)
            end = start + len(dfTest) - 1
            rmse, predictions = predictAndEvaluate(
                model, dfTest["Open"], title, start, end
            )
            if ar == 3 and ma == 2:
                showPredictedAndActual(dfTest["Open"], predictions, ar, ma)
            modelStats.append({"ar": ar, "ma": ma, "rmse": rmse})

    dfSolutions = pd.DataFrame(modelStats)
    dfSolutions = dfSolutions.sort_values(by=["rmse"])
    print(dfSolutions)


def ex3():
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np, pandas as pd
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.figsize": (9, 7), "figure.dpi": 120})

    # Import data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv",
        names=["value"],
        header=0,
    )
    df = df.diff()
    df = df.diff()
    print(df)
    df.value.plot()
    plt.title("www usage")
    plt.show()

    from statsmodels.tsa.stattools import adfuller

    result = adfuller(df.value.dropna())
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])


def main():
    # ex1()
    ex3()


if __name__ == "__main__":
    main()
