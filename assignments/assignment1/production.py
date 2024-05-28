import pandas as pd
import statsmodels.api as sm
import pickle

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)


def get_data():
    try:
        path = "/Users/elber/Documents/COMP 4949 - Datasets/4949_assignmentData.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df
    except FileNotFoundError:
        print("File not found")
        return
    except Exception as e:
        print(e)
        return


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


def main():
    try:
        columns = ["A", "F"]
        num_time_steps = 1
        df = get_data()
        df = prepare_data(df, columns, num_time_steps).dropna()

        features = df[["At-1", "Ft-1"]]
        features = sm.add_constant(features)

        file = open("best_model.pkl", "rb")
        model = pickle.load(file)

        # Make predictions.
        predictions = model.predict(features)

        print(f"Prediction for May 4, 2023: {predictions[-1]}")
    except FileNotFoundError:
        print("File not found")
        return
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    main()
