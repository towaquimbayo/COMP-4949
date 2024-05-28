def ex1():
    import pandas as pd

    co2 = [
        342.76,
        343.96,
        344.82,
        345.82,
        347.24,
        348.09,
        348.66,
        347.90,
        346.27,
        344.21,
        342.88,
        342.58,
        343.99,
        345.31,
        345.98,
        346.72,
        347.63,
        349.24,
        349.83,
        349.10,
        347.52,
        345.43,
        344.48,
        343.89,
        345.29,
        346.54,
        347.66,
        348.07,
        349.12,
        350.55,
        351.34,
        350.80,
        349.10,
        347.54,
        346.20,
        346.20,
        347.44,
        348.67,
    ]

    df = pd.DataFrame(
        {"CO2": co2},
        index=pd.date_range(start="09-01-2023", periods=len(co2), freq="W-MON"),
    )
    print(df)


def ex2():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until pandas_datareader is fixed.
    import pandas as pd
    import datetime

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

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

    NUM_DAYS = 10
    # Search Yahoo for the correct symbols.
    df = getStock("TD", NUM_DAYS)
    print("Toronto Dominion stock")
    print(df)


def ex3():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import pandas as pd
    import datetime

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

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

        if dtPast < datetime.date(2022, 1, 1):
            dtPast = datetime.date(2022, 1, 1)

        dtPastStr = dtPast.strftime("%Y-%m-%d")
        yfin.pdr_override()
        df = pdr.get_data_yahoo(stk, start=dtPastStr, end=dtNowStr)
        return df

    NUM_DAYS = 1000
    df = getStock("AMZN", NUM_DAYS)
    print("Amazon Stock Prices")
    print(df)

    import matplotlib.pyplot as plt

    def showStock(df, title):
        plt.plot(df.index, df["Close"])
        plt.title(title)
        plt.xticks(rotation=70)
        plt.show()

    showStock(df, "Amazon Stock Prices")


def ex4():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import pandas as pd
    import datetime

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

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

    import matplotlib.pyplot as plt

    def showStock(df, title):
        plt.plot(df.index, df["Close"])
        plt.title(title)
        plt.xticks(rotation=70)
        plt.show()

    # Get Southwestern stock for last 60 days
    NUM_DAYS = 1200
    df = getStock("LUV", NUM_DAYS)
    print("South West Airlines")
    print(df)

    # Create average monthly stock closing price at month end
    series = df["Close"].resample("M", convention="end").mean()
    summaryDf = series.to_frame()

    # Convert datetime index to date and then graph it.
    summaryDf.index = summaryDf.index.date
    print(summaryDf)
    showStock(summaryDf, "Weekly S.D. Southwest Airlines")


def ex5():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import pandas as pd
    import datetime

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

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

    import matplotlib.pyplot as plt

    def showStocks(df, stock, title):
        plt.plot(df.index, df["Close"], label=stock)
        plt.xticks(rotation=70)

    NUM_DAYS = 20
    df = getStock("AMZN", NUM_DAYS)
    df["Close"] = df["Close"].pct_change()
    showStocks(df, "AMZN", "AMZN Close Prices")

    df = getStock("AAPL", NUM_DAYS)
    df["Close"] = df["Close"].pct_change()
    showStocks(df, "AAPL", "AAPL Close Prices")

    df = getStock("MSFT", NUM_DAYS)
    df["Close"] = df["Close"].pct_change()
    showStocks(df, "MSFT", "MSFT Close Prices")

    # Make graphs appear.
    plt.legend()
    plt.show()


def ex6():
    import pandas as pd

    co2 = [342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27]
    df = pd.DataFrame(
        {"CO2": co2}, index=pd.date_range("09-01-2022", periods=len(co2), freq="B")
    )
    df["CO2_t-1"] = df["CO2"].shift(periods=1)
    df["CO2_t-2"] = df["CO2"].shift(periods=2)
    df = df.dropna()
    print(df)


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # ex5()
    ex6()


if __name__ == "__main__":
    main()
