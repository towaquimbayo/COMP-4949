import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic, adfuller
import pmdarima as pm
import pickle

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.options.mode.chained_assignment = None  # default='warn'


def get_data():
    path = "/Users/elber/Documents/COMP 4949 - Datasets/4949_assignmentData.csv"
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def check_missing_dates(df):
    expected_date_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="D"
    )
    missing_dates = expected_date_range.difference(df.index)
    print("Missing dates:")
    print(missing_dates)


def check_duplicate_dates(df):
    duplicate_dates = df.index[df.index.duplicated()]
    print("Duplicate Dates:", duplicate_dates)


def plot_feature(df, feature, percent_change=False):
    df_feature = df[feature]
    if percent_change:
        df_feature = df_feature.pct_change()
    plt.plot(df.index, df_feature)
    plt.title("Plot of feature: " + feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_seasonal_decompose(df, feature, method, show_components=False):
    tseries = seasonal_decompose(
        df[feature], model=method, extrapolate_trend="freq", period=12
    )

    tseries.plot()
    plt.tight_layout()
    plt.show()

    if not show_components:
        return
    plt.plot(tseries.trend)
    plt.title("Trend of Feature: " + feature, fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.plot(tseries.seasonal)
    plt.title("Seasonal of Feature: " + feature, fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.plot(tseries.resid)
    plt.title(
        f"{method.title()} Decomposed Residual of Feature: " + feature, fontsize=16
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_rolling_and_exp_smoothing(df, feature, days):
    rolling_mean = df[feature].rolling(window=days).mean()
    exp_smoothing = df[feature].ewm(span=days, adjust=False).mean()
    exp_smoothing_alpha = df[feature].ewm(alpha=0.4, adjust=False).mean()

    df[feature].plot(label="Original")
    rolling_mean.plot(label=f"Rolling Mean MA{days}")
    exp_smoothing.plot(label=f"Exponential Smoothing ES{days}")
    exp_smoothing_alpha.plot(label=f"Exponential Smoothing ES alpha=0.4")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, predictions, title="Actual vs Predicted Values"):
    plt.plot(y_test, label="Actual", marker="o")
    plt.plot(predictions, label="Predicted", marker="o")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# Create time shifted columns for as many time steps as specified
def back_shift_columns(df, original_col_name, num_time_steps):
    df_new = df[[original_col_name]].copy()  # .reset_index(drop=True)  # .pct_change()
    df_new.index = range(len(df_new))
    for i in range(1, num_time_steps + 1):
        new_col_name = original_col_name[0] + "t-" + str(i)
        df_new[new_col_name] = df_new[original_col_name].shift(periods=i)
    return df_new


def prepare_data(df, columns, num_time_steps):
    merged_df = pd.DataFrame()
    date_index = df.index
    for i in range(0, len(columns)):
        back_shifted_df = back_shift_columns(df, columns[i], num_time_steps)
        if i == 0:
            merged_df = back_shifted_df
        else:
            merged_df = merged_df.merge(
                back_shifted_df, left_index=True, right_index=True
            )
    merged_df.index = date_index
    return merged_df


def plot_corr_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr[["A"]].sort_values(by="A", ascending=False),
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="YlGnBu",
    )
    plt.title("Correlation Heatmap with Target Variable 'A'")
    plt.tight_layout()
    plt.show()

    # Correlation matrix
    pd.set_option("display.max_rows", None)  # Display all rows
    pd.set_option("display.max_columns", None)  # Display all columns
    pd.set_option("display.width", None)  # Allow unlimited width
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix["A"].sort_values(ascending=False))


def plot_single_holt_winters(df):
    df = df.copy()
    df["HWES1"] = (
        SimpleExpSmoothing(df["A"])
        .fit(smoothing_level=1 / (2 * 12), optimized=False, use_brute=True)
        .fittedvalues
    )
    df[["A", "HWES1"]].plot(title="Holt-Winters Exponential Smoothing")
    plt.tight_layout()
    plt.show()
    return df


def plot_double_holt_winters(df):
    df = df.copy()
    df["HWES2_ADD"] = ExponentialSmoothing(df["A"], trend="add").fit().fittedvalues
    df["HWES2_MUL"] = ExponentialSmoothing(df["A"], trend="mul").fit().fittedvalues
    df[["A", "HWES2_ADD", "HWES2_MUL"]].plot(
        title="Holt-Winters Double Exponential Smoothing: Add and Mult Trend"
    )
    plt.tight_layout()
    plt.show()
    return df


def plot_triple_holt_winters(df, period):
    df = df.copy()
    df["HWES3_ADD"] = (
        ExponentialSmoothing(
            df["A"], trend="add", seasonal="add", seasonal_periods=period
        )
        .fit()
        .fittedvalues
    )
    df["HWES3_MUL"] = (
        ExponentialSmoothing(
            df["A"], trend="mul", seasonal="mul", seasonal_periods=period
        )
        .fit()
        .fittedvalues
    )
    df[["A", "HWES3_ADD", "HWES3_MUL"]].plot(
        title="Holt-Winters Triple Exponential Smoothing: Add and Mult Seasonality"
    )
    plt.tight_layout()
    plt.show()
    return


def model_ols(df, test_days, num_time_steps):
    # Correlated (>= 0.5) features with target variable "A" (from correlation heatmap)
    # columns = ["A", "B", "D", "E", "F", "H", "R", "S", "U", "Z"]
    columns = ["A", "F"]  # Significant features

    # Prepare the data by back-shifting the features
    prepared_df = prepare_data(df, columns, num_time_steps)
    prepared_df = prepared_df.dropna()
    print("Dataframe Shape:", prepared_df.shape)
    print(prepared_df.head())
    plot_corr_heatmap(prepared_df)

    # Exponential Smoothing for all features (not needed as it provides no improvement)
    # exp_smoothed_df = prepared_df.copy()
    # for col in prepared_df.columns:
    #     exp_smoothed_col = prepared_df[col].ewm(alpha=0.3, adjust=False).mean()
    #     exp_smoothed_df[col + "_ES"] = exp_smoothed_col
    # print(exp_smoothed_df.head())
    # plot_corr_heatmap(exp_smoothed_df)

    # Assign the prepared dataframe with back-shifted features as main dataframe
    df = prepared_df

    features = [
        "At-1",  # Significant
        # "Bt-1",
        # "Ht-1",
        # "Dt-1",
        # "Ut-1",
        # "Et-1",
        # "At-2",
        # "Ht-2",
        # "Zt-1",
        # "Bt-2",
        "Ft-1",  # Significant
        # "Dt-2",
        # "St-1",
        # "Ut-2",
        # "Rt-1",
    ]
    # exp_features = [
    #     "At-1_ES",
    #     "Ht-1_ES",
    #     "Bt-1_ES",
    #     "Dt-1_ES",
    #     "Ut-1_ES",
    #     "Zt-1_ES",
    #     "Ft-1_ES",
    # ]
    # features.extend(exp_features)

    x = df[features]
    y = df[["A"]]
    x = sm.add_constant(x)

    # Split into test and train sets. The test data must be the latest data range.
    len_data = len(df)
    x_train = x[0 : len_data - test_days]
    y_train = y[0 : len_data - test_days]
    x_test = x[len_data - test_days :]
    y_test = y[len_data - test_days :]

    # Model and make predictions.
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)
    print(model.summary())

    # Fix the index of the predictions
    predictions = pd.Series(predictions)
    predictions.index = y_test.index

    # Show RMSE and plot the data.
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

    # Plot the actual vs predicted values
    plot_actual_vs_predicted(y_test, predictions)

    # Save model as binary pickle file
    # x = prepared_df[features]
    # x = sm.add_constant(x)
    # y = prepared_df[["A"]]
    # model = sm.OLS(y, x).fit()
    #
    # file_handler = open("best_model.pkl", "wb")
    # pickle.dump(model, file_handler)
    # file_handler.close()


def model_holt_winters(df, test_days):
    # Drop all columns except "A" the target variable
    df = df.drop(columns=[col for col in df.columns if col != "A"])

    # df.index.freq = "MS"

    # Single Holt-Winters Exponential Smoothing (HWES1)
    # df = plot_single_holt_winters(df)

    # Double Holt-Winters Exponential Smoothing (HWES2)
    # df = plot_double_holt_winters(df)

    # Triple Holt-Winters Exponential Smoothing (HWES3)
    # df = plot_triple_holt_winters(df, 12)

    # Split into test and train sets. The test data must be the latest data range.
    train = df[0 : len(df) - test_days]
    test = df[len(df) - test_days :]

    # Build HWES3 model with multiplicative decomposition
    model = ExponentialSmoothing(
        train["A"], trend="mul", seasonal="mul", seasonal_periods=12
    ).fit()
    predictions = model.forecast(test_days)

    # Fix the index of the predictions
    predictions.index = test.index
    print(predictions)

    # Show RMSE
    print("RMSE:", np.sqrt(mean_squared_error(test["A"], predictions)))

    # Plot the actual vs predicted values
    plot_actual_vs_predicted(test["A"], predictions)

    # Plot raw train, test and predicted values
    train["A"].plot(legend=True, label="Train")
    test["A"].plot(legend=True, label="Test")
    predictions.plot(legend=True, label="Predictions")
    plt.title("Train, Test and Predicted Values of 'A' using HWES3 Multiplicative")
    plt.tight_layout()
    plt.show()


def model_ar(df, test_days, num_time_steps):
    # Show auto-correlation (ACF) and partial auto-correlation (PACF) plots
    # plot_acf(df["A"], lags=50)
    # plot_pacf(df["A"], lags=50)
    # plt.show()
    # plot_acf(df["F"], lags=50)
    # plot_pacf(df["F"], lags=50)
    # plt.show()

    # Split into test and train sets. The test data must be the latest data range.
    train = df[0 : len(df) - test_days]
    test = df[len(df) - test_days :]

    def build_autoreg_model(df_train, df_test):
        model = AutoReg(df_train, lags=num_time_steps)
        model_fit = model.fit()

        print("Coefficients:", model_fit.params)
        print(model_fit.summary())

        # Make predictions
        predictions = model_fit.predict(
            start=len(df_train), end=len(df_train) + len(df_test) - 1, dynamic=False
        )

        # Fix the index of the predictions
        predictions.index = df_test.index

        for i in range(len(predictions)):
            print(f"predicted={predictions[i]}, expected={df_test[i]}")
        print("RMSE:", np.sqrt(mean_squared_error(df_test, predictions)))
        print("Model AIC:", model_fit.aic)
        print("Model BIC:", model_fit.bic)

        # Plot the actual vs predicted values
        plot_actual_vs_predicted(df_test, predictions)

    def build_ar_arima_model(df_train, df_test):
        model = ARIMA(df_train, order=(num_time_steps, 0, 0)).fit()

        print(f"Evaluating ARMA({num_time_steps}, 0, 0) Model")
        print("Coefficients:", model.params)
        print(model.summary())

        # Make predictions
        predictions = model.predict(
            start=len(df_train), end=len(df_train) + len(df_test) - 1, dynamic=False
        )

        # Fix the index of the predictions
        predictions.index = df_test.index

        print("RMSE:", np.sqrt(mean_squared_error(df_test, predictions)))
        print("Model AIC:", model.aic)
        print("Model BIC:", model.bic)

        # Plot the actual vs predicted values
        plot_actual_vs_predicted(df_test, predictions)

    # Using model coefficients from ARMA model to make future predictions
    def make_prediction(t_1, t_2):
        intercept = 336.463902
        t1_coeff = 0.889947
        t2_coeff = -0.107334

        predictions = intercept + t1_coeff * t_1 + t2_coeff * t_2
        return predictions

    res = arma_order_select_ic(df["A"], max_ar=10, max_ma=0, ic=["aic", "bic"])
    print(res.values())

    # Build AR model using AutoReg. RMSE: 398.5016823817488
    build_autoreg_model(train["A"], test["A"])

    # Build AR model using ARIMA. RMSE: 398.5016823817488
    build_ar_arima_model(train["A"], test["A"])

    # Make future prediction using AR model coefficients for the next 2 days
    prediction_days = 2  # Number of days to predict
    t_1 = test["A"].iloc[-1]
    t_2 = test["A"].iloc[-2]

    future_predictions = []
    for i in range(0, prediction_days):
        prediction = make_prediction(t_1, t_2)
        future_predictions.append(prediction)
        t_2 = t_1
        t_1 = prediction

    print("Future Predictions for the next 2 days:")
    print(future_predictions)


def model_arma(df, test_days):
    # Split into test and train sets. The test data must be the latest data range.
    train = df[0 : len(df) - test_days]
    test = df[len(df) - test_days :]

    # Grid search for ARMA parameters (AR, MA)
    model_stats = []
    for ar in range(0, 5):
        for ma in range(0, 5):
            model = ARIMA(train["A"], order=(ar, 0, ma)).fit()
            print(f"\n*** {ar}_0_{ma} ***")
            print(model.summary())
            predictions = model.predict(
                start=len(train), end=len(train) + len(test) - 1, dynamic=True
            )

            # Fix the index of the predictions
            predictions.index = test.index

            rmse = np.sqrt(mean_squared_error(test["A"], predictions))
            print(f"RMSE: {rmse}")

            # Best ARMA model: AR:3, MA:4
            if ar == 3 and ma == 4:
                plot_actual_vs_predicted(
                    test["A"],
                    predictions,
                    f"Actual vs Predicted Values: AR:{ar} MA:{ma}",
                )
            model_stats.append({"ar": ar, "ma": ma, "rmse": rmse})

    df_solutions = pd.DataFrame(model_stats).sort_values(by=["rmse"])
    print(df_solutions)


def model_arima_with_dif(df, test_days):
    df = df.drop(columns=[col for col in df.columns if col != "A"])
    # Split into test and train sets. The test data must be the latest data range.
    x_train = df[0 : len(df) - test_days]
    y_train = df[0 : len(df) - test_days]
    x_test = df[len(df) - test_days :]
    y_test = df[len(df) - test_days :]

    # Create a list with the training array.
    predictions = []

    for i in range(len(x_test)):
        print("History length:", len(x_train))

        # Build and predict model using ARIMA
        model = ARIMA(x_train, order=(1, 1, 1)).fit()
        yhat = model.predict(start=len(x_train), end=len(x_train))

        # print(model.summary())

        if i < len(x_test):
            test_row = x_test.iloc[i]
            x_train = x_train._append(test_row, ignore_index=True)
            predictions.append(yhat.iloc[0])
        else:
            break

    # Fix the index of the predictions
    predictions = pd.Series(predictions)
    predictions.index = x_test.index

    # Show RMSE and plot the data.
    print("RMSE:", np.sqrt(mean_squared_error(x_test, predictions)))
    plot_actual_vs_predicted(
        x_test, predictions, "Actual vs Predicted Values: ARIMA(1, 1, 1)"
    )


def model_auto_arima(df, test_days):
    df = df.drop(columns=[col for col in df.columns if col != "A"])

    # Split into test and train sets. The test data must be the latest data range.
    train = df[0 : len(df) - test_days]
    test = df[len(df) - test_days :]

    # Create a list with the training array.
    predictions = []

    for i in range(len(test)):
        print("\n********************")
        print("Iteration:", i)
        print("Length of training data:", len(train))
        d = ndiffs(train, test="adf")  # Find the number of differences needed
        model = pm.auto_arima(
            train,
            start_p=1,
            start_q=1,
            d=d,
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

        yhat = model.predict(start=len(train), end=len(train))

        if i < len(test):
            test_row = test.iloc[i]
            train = train._append(test_row, ignore_index=True)
            predictions.append(yhat.iloc[0])
        else:
            break

    # Fix the index of the predictions
    predictions = pd.Series(predictions)
    predictions.index = test.index

    # Show RMSE and plot the data.
    print("RMSE:", np.sqrt(mean_squared_error(test, predictions)))
    plot_actual_vs_predicted(
        test, predictions, "Actual vs Predicted Values: SARIMAX(4, 0, 5)"
    )


def main():
    df = get_data()

    # Check for missing dates
    # check_missing_dates(df)

    # Check for duplicate dates
    # check_duplicate_dates(df)

    # print(df.info)
    # print(df.index)
    # print(df.index.freq)
    # print(df.shape)
    # print(df.head())
    # print(df.isnull().sum())
    # print(df.pct_change())

    # Plot the features of the dataframe
    # for col in df.columns:
    #     if col in ["A", "F"]:
    #         plot_feature(df, col, False)

    # Perform decomposition using additive/multiplicative decomposition.
    # for col in df.columns:
    #     if col in ["A", "F"]:
    #         plot_seasonal_decompose(df, col, "multiplicative", True)

    # for col in df.columns:
    #     if col in ["A", "F"]:
    #         plot_seasonal_decompose(df, col, "additive", True)

    # Plot the rolling mean and exponential smoothing
    # for col in df.columns:
    #     plot_rolling_and_exp_smoothing(df, col, 50)

    test_days = 10
    num_time_steps = 2

    # OLS Linear Regression Model (Best Model)
    model_ols(df, test_days, num_time_steps)

    # Holt Winter's Model
    # model_holt_winters(df, test_days)

    # AR Model
    # model_ar(df, test_days, num_time_steps)

    # ARMA Model
    # model_arma(df, test_days)

    # ARIMA Model with Differencing
    # model_arima_with_dif(df, test_days)

    # ARIMA Model using AutoARIMA to find optimal parameters for p, d, q
    # model_auto_arima(df, test_days)


if __name__ == "__main__":
    main()
