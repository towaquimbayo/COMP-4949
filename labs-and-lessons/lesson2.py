def ex1():
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt

    # Import data.
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE = "drugSales.csv"
    df = pd.read_csv(PATH + FILE, parse_dates=["date"], index_col="date")
    type(df.index)

    # Perform decomposition using multiplicative decomposition.
    tseries = seasonal_decompose(
        df["value"], model="multiplicative", extrapolate_trend="freq"
    )

    tseries.plot()
    plt.show()

    # Extract the Components ----
    # Actual Values = Product of (Seasonal * Trend * Resid)
    dfComponents = pd.concat(
        [tseries.seasonal, tseries.trend, tseries.resid, tseries.observed], axis=1
    )
    dfComponents.columns = ["seas", "trend", "resid", "actual_values"]
    print(dfComponents.head())

    def calculate_component(index):
        return tseries.seasonal[index] * tseries.trend[index] * tseries.resid[index]

    print("Actual Values of Second Row:")
    print(calculate_component(1))


def ex2():
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt

    # Import data.
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE = "AirPassengers.csv"
    df = pd.read_csv(PATH + FILE, parse_dates=["date"], index_col="date")
    type(df.index)

    # Perform decomposition using multiplicative decomposition.
    tseries_add = seasonal_decompose(
        df["value"], model="additive", extrapolate_trend="freq"
    )
    tseries_mult = seasonal_decompose(
        df["value"], model="multiplicative", extrapolate_trend="freq"
    )

    tseries_add.plot()
    tseries_mult.plot()
    plt.show()


def ex3():
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt

    # Import Data
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE = "AirPassengers.csv"
    df = pd.read_csv(PATH + FILE, parse_dates=["date"], index_col="date")
    tseries = seasonal_decompose(
        df["value"], model="additive", extrapolate_trend="freq"
    )

    plt.plot(df["value"])
    plt.title("Airplane Passengers", fontsize=16)
    plt.show()

    detrended = df["value"] - tseries.trend
    plt.plot(detrended)
    plt.title("Airplane Passengers After Subtracting Trend", fontsize=16)
    plt.show()


def ex4():
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt

    # Import Data
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE = "AirPassengers.csv"
    df = pd.read_csv(PATH + FILE, parse_dates=["date"], index_col="date")
    tseries = seasonal_decompose(
        df["value"], model="multiplicative", extrapolate_trend="freq"
    )

    plt.plot(df["value"])
    plt.title("Airplane Passengers", fontsize=16)
    plt.show()

    deseasonalized = df.value.values / tseries.seasonal
    plt.plot(deseasonalized)
    plt.title("Airplane Passengers After De-Seasonalizing", fontsize=16)
    plt.show()


def ex8():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import datetime
    import matplotlib.pyplot as plt

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

    df = getStock("MSFT", 200)

    # Calculating the moving averages.
    rolling_mean = df["Close"].rolling(window=50).mean()

    # Calculate the exponentially smoothed series.
    exp50 = df["Close"].ewm(span=50, adjust=False).mean()

    df["Close"].plot(label="MSFT Close ", color="gray", alpha=0.3)
    rolling_mean.plot(label="MSFT 50 Day MA", style="--", color="orange")
    exp50.plot(label="MSFT 50 Day ES", style="--", color="green")
    plt.legend()
    plt.show()


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    ex8()


if __name__ == "__main__":
    main()
