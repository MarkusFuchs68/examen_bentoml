import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import pickle


def load_data():
    # Load the dataset
    data_path = os.path.join(os.getcwd(), "data/processed")

    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model


def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

    # evaluate the model using mean squared error
    y_test = y_test.values.ravel()  # Ensure y_test is a 1D array
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error: {rmse}")

    return score, mse, rmse


def save_model(model):
    # Save the model
    model_path = os.path.join(os.getcwd(), "models")
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "admission_rf.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")


def register_model(model, score, mse, rmse):
    # Register the model with BentoML
    import bentoml
    # Save the model to BentoML
    bento_model = bentoml.sklearn.save_model(
        "admission_rf",
        model,
        labels={"model_type": "RandomForestRegressor"},
        metadata={
            "description": "Random Forest Regressor for admission prediction",
            "metrics":{
                "R^2": score,
                "MSE": mse,
                "RMSE": rmse,
            },
        },
    )
    print(f"Model registered with BentoML: {bento_model}")


if __name__ == "__main__":
    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    score, mse, rmse = evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model)

    # Register the model with BentoML
    register_model(model, score, mse, rmse)
