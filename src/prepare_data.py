import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the data from the csv file
    :return: pandas dataframe
    """
    file_path = os.path.join(os.getcwd(), 'data/raw/admission.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    df = pd.read_csv(file_path, index_col=0)

    # check for missing values, there should be none
    if df.isnull().sum().sum() > 0:
        raise ValueError("Data contains missing values")

    return df


def separate_split_data(df, test_size=0.2):
    """
    Split the data into train and test sets
    :param df: pandas dataframe
    :param test_size: size of the test set
    :return: normalized train and test sets and labels
    """
    # separate the features and labels
    X = df.drop(columns=['Chance of Admit '])
    y = df['Chance of Admit ']

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # since we are using a RandomForestRegressor, we don't need to normalize the data
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    """
    Save the data to the data folder
    :param X_train: train set
    :param X_test: test set
    :param y_train: train labels
    :param y_test: test labels
    """
    preprocess_folder = os.path.join(os.getcwd(), 'data/processed')
    os.makedirs(preprocess_folder, exist_ok=True)

    X_train.to_csv(os.path.join(preprocess_folder, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(preprocess_folder, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(preprocess_folder, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(preprocess_folder, 'y_test.csv'), index=False)


if __name__ == '__main__':
    # load the data
    df = load_data()

    # separate, split and normalize the data
    X_train, X_test, y_train, y_test = separate_split_data(df)

    # save the data
    save_data(X_train, X_test, y_train, y_test)
    print("Data preparation complete. Data saved to data/processed folder.")
