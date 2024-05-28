import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from sklearn.metrics import (
    classification_report,
    precision_score,
    f1_score,
    recall_score,
)
import warnings

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
warnings.filterwarnings("ignore")


def get_data():
    return pd.read_csv("./train.csv")


def prepare_data(df):
    # Drop 'employee_id' because it's a unique identifier
    # Drop 'region' because it's a categorical variable with 34 unique values
    df = df.drop(["employee_id", "region"], axis=1)

    # Impute missing values in 'previous_year_rating' and 'education'
    df = treat_missing_values(df)

    # Treat outliers
    df = treat_outliers(df)

    # Label encode 'department'
    df["department"] = df["department"].astype("category").cat.codes

    # Label encode 'education'
    df["education"] = df["education"].astype("category").cat.codes

    # Label encode 'gender'
    df["gender"] = df["gender"].astype("category").cat.codes

    # Label encode 'recruitment_channel'
    df["recruitment_channel"] = df["recruitment_channel"].astype("category").cat.codes

    # Check data after preprocessing
    print("Preprocessed Data:")
    print(df.head())
    return df


# Impute missing values
def treat_missing_values(df):
    # Fill 'previous_year_rating' with 0
    df["previous_year_rating"] = df["previous_year_rating"].fillna(0)

    # Fill 'education' with Mode values because it's a categorical ordinal variable
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    return df


def treat_outliers(df):
    # Treat 'length_of_service' outliers by capping values greater than 13 (Q3)
    df["length_of_service"] = np.where(
        df["length_of_service"] > 13, 13, df["length_of_service"]
    )
    return df


# Artificial Neural Network
def model_ann(df):
    df = df[
        [
            "department",
            "previous_year_rating",
            "awards_won?",
            "avg_training_score",
            "is_promoted",
        ]
    ]
    # Split the data into features and target
    dataset = df.values
    x = dataset[:, 0 : df.shape[1] - 1]
    y = dataset[:, df.shape[1] - 1]
    ROW_DIM = 0
    COL_DIM = 1

    x_array_reshaped = x.reshape(x.shape[ROW_DIM], x.shape[COL_DIM])
    y_array_reshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data into training and testing sets
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_array_reshaped, y_array_reshaped, test_size=0.3
    )
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)

    # Save the scaler
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    def create_model(
        optimizer,
        learning_rate=0.001,
        initializer="normal",
        neurons=12,
        additional_layers=0,
        activation_function="relu",
    ):
        # Build an Artificial Neural Network model of sequential layers
        model = Sequential()

        # Add first hidden layer (input layer)
        model.add(
            Dense(
                neurons,
                input_dim=x_train.shape[1],
                activation=activation_function,
                kernel_initializer=initializer,
            )
        )

        # Add additional hidden layers
        for i in range(additional_layers):
            model.add(
                Dense(
                    neurons,
                    activation=activation_function,
                    kernel_initializer=initializer,
                )
            )

        # Add output layer
        model.add(Dense(1, kernel_initializer=initializer, activation="sigmoid"))

        # Compile the model
        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def evaluate_model(model, x_test, y_test):
        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("\n*** Evaluate Artificial Neural Network Model ***")
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        # Make predictions for the test data
        predictions = model.predict(x_test)
        print("Actual:")
        print(y_test)
        print("Predictions:")
        print(predictions)

        # Convert predictions to binary values and print classification report
        predictions = [1 if x > 0.5 else 0 for x in predictions]
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return loss, accuracy, precision, recall, f1, predictions

    BEST_EPOCHS = 50
    BEST_BATCH_SIZE = 100
    BEST_OPTIMIZER = RMSprop
    BEST_LEARNING_RATE = 0.01
    BEST_INITIALIZER = "he_normal"
    BEST_NEURONS = 150
    BEST_ADDITIONAL_LAYERS = 3
    BEST_ACTIVATION_FUNCTION = "softsign"

    # Build the stand-alone ANN model
    print("\n*** Stand-Alone Model Evaluation ***")
    model = create_model(
        BEST_OPTIMIZER,
        BEST_LEARNING_RATE,
        BEST_INITIALIZER,
        BEST_NEURONS,
        BEST_ADDITIONAL_LAYERS,
        BEST_ACTIVATION_FUNCTION,
    )

    # Create early stopping and model checkpoint callbacks
    es = EarlyStopping(
        monitor="val_loss", min_delta=0.000001, mode="min", verbose=1, patience=20
    )
    mc = ModelCheckpoint(
        "best_model.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    # Fit the model
    model.fit(
        x_train,
        y_train,
        epochs=BEST_EPOCHS,
        batch_size=BEST_BATCH_SIZE,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[es, mc],
    )

    # Evaluate the model
    evaluate_model(model, x_test, y_test)


def main():
    # Get the data
    df = get_data()

    # Prepare data that imputes missing values and treats outliers
    df = prepare_data(df)

    # Model: Artificial Neural Network
    model_ann(df)


if __name__ == "__main__":
    main()
