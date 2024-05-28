def ex1():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
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

    def getNewBalance(startBalance, startPrice, endPrice):
        qty = int(startBalance / startPrice)
        cashLeftOver = startBalance - qty * startPrice
        endValue = qty * endPrice
        balance = cashLeftOver + endValue
        return balance

    def showBuyAndHoldEarnings(df, balance):
        startClosePrice = df.iloc[0]["Close"]
        endClosePrice = df.iloc[len(df) - 1]["Close"]
        newBalance = getNewBalance(balance, startClosePrice, endClosePrice)
        print("Buy and hold closing balance: $" + str(round(newBalance, 2)))

    def showStrategyEarnings(df, balance, lt, st):
        buyPrice = 0
        buyDate = None
        sellDate = None
        bought = False

        buySellDates = []
        prices = []

        dfStrategy = pd.DataFrame(
            columns=["buyDt", "buy$", "sellDt", "sell$", "balance"]
        )
        dates = list(df.index)
        for i in range(0, len(df)):
            if df.iloc[i]["Buy"] and not bought:
                buyPrice = df.iloc[i]["Close"]
                buyDate = dates[i]
                bought = True
                buySellDates.append(buyDate)
                prices.append(buyPrice)

            elif df.iloc[i]["Sell"] and bought:
                sellPrice = df.iloc[i]["Close"]
                balance = getNewBalance(balance, buyPrice, sellPrice)
                sellDate = dates[i]
                buySellInfo = {
                    "buyDt": buyDate,
                    "buy$": buyPrice,
                    "sellDt": sellDate,
                    "sell$": sellPrice,
                    "balance": balance,
                }
                dfStrategy = dfStrategy.append(buySellInfo, ignore_index=True)
                bought = False
                buySellDates.append(sellDate)
                prices.append(sellPrice)

        print(dfStrategy)
        print("\nMoving average strategy closing balance: $" + str(round(balance, 2)))
        return buySellDates, prices

    def showBuyAndSellDates(df, startBalance):
        strategyDates, strategyPrices = showStrategyEarnings(df, startBalance, lt, st)
        plt.plot(df.index, df["Close"], label="Close")
        plt.plot(df.index, df["ema20"], label="ema20", alpha=0.4)
        plt.plot(df.index, df["ema50"], label="ema50", alpha=0.4)
        plt.scatter(strategyDates, strategyPrices, label="Buy/Sell", color="red")
        plt.xticks(rotation=70)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def showInvestmentDifferences(dfStock, lt, st):
        df = dfStock.copy()
        df["ema50"] = df["Close"].ewm(span=lt).mean()
        df["ema20"] = df["Close"].ewm(span=st).mean()

        # Remove nulls.
        df.dropna(inplace=True)
        df.round(3)
        own_positions = np.where(df["ema20"] > df["ema50"], 1, 0)
        df["Position"] = own_positions
        df.round(3)

        df["Buy"] = (df["Position"] == 1) & (df["Position"].shift(1) == 0)
        df["Sell"] = (df["Position"] == 0) & (df["Position"].shift(1) == 1)

        START_BALANCE = 10000

        print("-------------------------------------------------------")
        showBuyAndHoldEarnings(df, START_BALANCE)
        print("-------------------------------------------------------")
        showBuyAndSellDates(df, START_BALANCE)

    longterms = [50, 100, 200]
    shortterms = [20, 30, 50]
    dfStock = getStock("AMD", 1100)

    for lt in longterms:
        for st in shortterms:
            print("\b******************************************************")
            print("Lt: " + str(lt))
            print("St: " + str(st))
            showInvestmentDifferences(dfStock, lt, st)


def ex2():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
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

    def getNewBalance(startBalance, startPrice, endPrice):
        qty = int(startBalance / startPrice)
        cashLeftOver = startBalance - qty * startPrice
        endValue = qty * endPrice
        balance = cashLeftOver + endValue
        return balance

    def showBuyAndHoldEarnings(df, balance):
        startClosePrice = df.iloc[0]["Close"]
        endClosePrice = df.iloc[len(df) - 1]["Close"]
        newBalance = getNewBalance(balance, startClosePrice, endClosePrice)
        print("Buy and hold closing balance: $" + str(round(newBalance, 2)))

    def showStrategyEarnings(df, balance, lt, st):
        buyPrice = 0
        buyDate = None
        sellDate = None
        bought = False

        buySellDates = []
        prices = []

        dfStrategy = pd.DataFrame(
            columns=["buyDt", "buy$", "sellDt", "sell$", "balance"]
        )
        dates = list(df.index)
        for i in range(0, len(df)):
            if df.iloc[i]["Buy"] and not bought:
                buyPrice = df.iloc[i]["Close"]
                buyDate = dates[i]
                bought = True
                buySellDates.append(buyDate)
                prices.append(buyPrice)

            elif df.iloc[i]["Sell"] and bought:
                sellPrice = df.iloc[i]["Close"]
                balance = getNewBalance(balance, buyPrice, sellPrice)
                sellDate = dates[i]
                buySellInfo = {
                    "buyDt": buyDate,
                    "buy$": buyPrice,
                    "sellDt": sellDate,
                    "sell$": sellPrice,
                    "balance": balance,
                }
                dfStrategy = dfStrategy.append(buySellInfo, ignore_index=True)
                bought = False
                buySellDates.append(sellDate)
                prices.append(sellPrice)

        print(dfStrategy)
        print("\nMoving average strategy closing balance: $" + str(round(balance, 2)))
        return buySellDates, prices

    def showBuyAndSellDates(df, startBalance):
        strategyDates, strategyPrices = showStrategyEarnings(df, startBalance, lt, st)
        plt.plot(df.index, df["Close"], label="Close")
        plt.plot(df.index, df["ema20"], label="ema20", alpha=0.4)
        plt.plot(df.index, df["ema50"], label="ema50", alpha=0.4)
        plt.scatter(strategyDates, strategyPrices, label="Buy/Sell", color="red")
        plt.xticks(rotation=70)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def showInvestmentDifferences(dfStock, lt, st):
        df = dfStock.copy()
        df["ema50"] = df["Close"].ewm(span=lt).mean()
        df["ema20"] = df["Close"].ewm(span=st).mean()

        # Remove nulls.
        df.dropna(inplace=True)
        df.round(3)
        own_positions = np.where(df["ema20"] > df["ema50"], 1, 0)
        df["Position"] = own_positions
        df.round(3)

        df["Buy"] = (df["Position"] == 1) & (df["Position"].shift(1) == 0)
        df["Sell"] = (df["Position"] == 0) & (df["Position"].shift(1) == 1)

        START_BALANCE = 10000

        print("-------------------------------------------------------")
        showBuyAndHoldEarnings(df, START_BALANCE)
        print("-------------------------------------------------------")
        showBuyAndSellDates(df, START_BALANCE)

    longterms = [50, 100, 200]
    shortterms = [20, 30, 50]
    dfStock = getStock("XOM", 1100)

    for lt in longterms:
        for st in shortterms:
            print("\b******************************************************")
            print("Lt: " + str(lt))
            print("St: " + str(st))
            showInvestmentDifferences(dfStock, lt, st)


def ex4():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd

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

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Do not show warning.
    pd.options.mode.chained_assignment = None  # default='warn'

    ##################################################################
    # CONFIGURATION SECTION
    NUM_DAYS = 1200
    NUM_TIME_STEPS = 2
    TEST_DAYS = 30

    ##################################################################
    # Creates time shifted columns for as many time steps needed.
    def backShiftColumns(df, originalColName, numTimeSteps):
        dfNew = df[[originalColName]].pct_change()

        for i in range(1, numTimeSteps + 1):
            newColName = originalColName[0] + "t-" + str(i)
            dfNew[newColName] = dfNew[originalColName].shift(periods=i)
        return dfNew

    def prepareStockDf(stockSymbol, columns):
        df = getStock(stockSymbol, NUM_DAYS)

        # Create data frame with back shift columns for all features of interest.
        mergedDf = pd.DataFrame()
        for i in range(0, len(columns)):
            backShiftedDf = backShiftColumns(df, columns[i], NUM_TIME_STEPS)
            if i == 0:
                mergedDf = backShiftedDf
            else:
                mergedDf = mergedDf.merge(
                    backShiftedDf, left_index=True, right_index=True
                )

        newColumns = list(mergedDf.keys())

        # Append stock symbol to column names.
        for i in range(0, len(newColumns)):
            mergedDf.rename(
                columns={newColumns[i]: stockSymbol + "_" + newColumns[i]}, inplace=True
            )

        return mergedDf

    columns = ["Open", "Close"]
    msftDf = prepareStockDf("MSFT", columns)
    aaplDf = prepareStockDf("AAPL", columns)
    mergedDf = msftDf.merge(aaplDf, left_index=True, right_index=True)
    mergedDf = mergedDf.dropna()
    print(mergedDf)

    import seaborn as sns

    corr = mergedDf.corr()
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        corr[["MSFT_Open"]].sort_values(by="MSFT_Open", ascending=False),
        linewidth=0.5,
        vmin=-1,
        annot=True,
        vmax=1,
        cmap="YlGnBu",
    )
    plt.tight_layout()
    plt.show()


def ex6():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd

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

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Do not show warning.
    pd.options.mode.chained_assignment = None  # default='warn'

    ##################################################################
    # CONFIGURATION SECTION
    NUM_DAYS = 1200
    NUM_TIME_STEPS = 2
    TEST_DAYS = 30

    ##################################################################
    # Creates time shifted columns for as many time steps needed.
    def backShiftColumns(df, originalColName, numTimeSteps):
        dfNew = df[[originalColName]]

        for i in range(1, numTimeSteps + 1):
            newColName = originalColName[0] + "t-" + str(i)
            dfNew[newColName] = dfNew[originalColName].shift(periods=i)
        return dfNew

    def prepareStockDf(stockSymbol, columns):
        df = getStock(stockSymbol, NUM_DAYS)

        # Create data frame with back shift columns for all features of interest.
        mergedDf = pd.DataFrame()
        for i in range(0, len(columns)):
            backShiftedDf = backShiftColumns(df, columns[i], NUM_TIME_STEPS)
            if i == 0:
                mergedDf = backShiftedDf
            else:
                mergedDf = mergedDf.merge(
                    backShiftedDf, left_index=True, right_index=True
                )

        newColumns = list(mergedDf.keys())

        # Append stock symbol to column names.
        for i in range(0, len(newColumns)):
            mergedDf.rename(
                columns={newColumns[i]: stockSymbol + "_" + newColumns[i]}, inplace=True
            )

        return mergedDf

    columns = ["Open", "Close"]
    msftDf = prepareStockDf("MSFT", columns)
    aaplDf = prepareStockDf("AAPL", columns)
    mergedDf = msftDf.merge(aaplDf, left_index=True, right_index=True)
    mergedDf = mergedDf.dropna()
    print(mergedDf)

    import seaborn as sns

    corr = mergedDf.corr()
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        corr[["MSFT_Open"]].sort_values(by="MSFT_Open", ascending=False),
        linewidth=0.5,
        vmin=-1,
        annot=True,
        vmax=1,
        cmap="YlGnBu",
    )
    plt.tight_layout()
    plt.show()

    xfeatures = [
        "MSFT_Ot-1",
        "MSFT_Ot-2",
        "MSFT_Ct-1",
        "MSFT_Ct-2",
        "AAPL_Ot-1",
        "AAPL_Ot-2",
        "AAPL_Ct-1",
        "AAPL_Ct-2",
    ]
    X = mergedDf[xfeatures]
    y = mergedDf[["MSFT_Open"]]

    # Add intercept for OLS regression.
    import statsmodels.api as sm

    X = sm.add_constant(X)

    # Split into test and train sets. The test data must be
    # the latest data range.
    lenData = len(X)
    X_train = X[0 : lenData - TEST_DAYS]
    y_train = y[0 : lenData - TEST_DAYS]
    X_test = X[lenData - TEST_DAYS :]
    y_test = y[lenData - TEST_DAYS :]

    # Model and make predictions.
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())
    predictions = model.predict(X_test)

    # Show RMSE and plot the data.
    from sklearn import metrics
    import numpy as np

    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )

    plt.plot(y_test, label="Actual", marker="o")
    plt.plot(predictions, label="Predicted", marker="o")
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def ex7():
    from pandas_datareader import data as pdr
    import yfinance as yfin  # Work around until

    # pandas_datareader is fixed.
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd

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

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Do not show warning.
    pd.options.mode.chained_assignment = None  # default='warn'

    ##################################################################
    # CONFIGURATION SECTION
    NUM_DAYS = 1200
    NUM_TIME_STEPS = 2
    TEST_DAYS = 30

    ##################################################################
    # Creates time shifted columns for as many time steps needed.
    def backShiftColumns(df, originalColName, numTimeSteps):
        dfNew = df[[originalColName]].pct_change()

        for i in range(1, numTimeSteps + 1):
            newColName = originalColName[0] + "t-" + str(i)
            dfNew[newColName] = dfNew[originalColName].shift(periods=i)
        return dfNew

    def prepareStockDf(stockSymbol, columns):
        df = getStock(stockSymbol, NUM_DAYS)

        # Create data frame with back shift columns for all features of interest.
        mergedDf = pd.DataFrame()
        for i in range(0, len(columns)):
            backShiftedDf = backShiftColumns(df, columns[i], NUM_TIME_STEPS)
            if i == 0:
                mergedDf = backShiftedDf
            else:
                mergedDf = mergedDf.merge(
                    backShiftedDf, left_index=True, right_index=True
                )

        newColumns = list(mergedDf.keys())

        # Append stock symbol to column names.
        for i in range(0, len(newColumns)):
            mergedDf.rename(
                columns={newColumns[i]: stockSymbol + "_" + newColumns[i]}, inplace=True
            )

        return mergedDf

    columns = ["Open", "Close"]
    msftDf = prepareStockDf("MSFT", columns)
    aaplDf = prepareStockDf("GOOGL", columns)
    mergedDf = msftDf.merge(aaplDf, left_index=True, right_index=True)
    mergedDf = mergedDf.dropna()
    print(mergedDf)

    import seaborn as sns

    corr = mergedDf.corr()
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        corr[["MSFT_Open"]].sort_values(by="MSFT_Open", ascending=False),
        linewidth=0.5,
        vmin=-1,
        annot=True,
        vmax=1,
        cmap="YlGnBu",
    )
    plt.tight_layout()
    plt.show()

    xfeatures = ["MSFT_Ct-1"]
    X = mergedDf[xfeatures]
    y = mergedDf[["MSFT_Open"]]

    # Add intercept for OLS regression.
    import statsmodels.api as sm

    X = sm.add_constant(X)

    # Split into test and train sets. The test data must be
    # the latest data range.
    lenData = len(X)
    X_train = X[0 : lenData - TEST_DAYS]
    y_train = y[0 : lenData - TEST_DAYS]
    X_test = X[lenData - TEST_DAYS :]
    y_test = y[lenData - TEST_DAYS :]

    # Model and make predictions.
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())
    predictions = model.predict(X_test)

    # Show RMSE and plot the data.
    from sklearn import metrics
    import numpy as np

    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )

    plt.plot(y_test, label="Actual", marker="o")
    plt.plot(predictions, label="Predicted", marker="o")
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    # ex1()
    # ex2()
    # ex4()
    # ex6()
    ex7()


if __name__ == "__main__":
    main()
