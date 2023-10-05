import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataFrameException(Exception):
    """
    Custom exception class for handling DataFrame-related errors.
    """

    def __init__(self, message = "Custom Exception"):
        super().__init__(message)

class dataloader:
    """
    Data loading and preprocessing class for the Iris dataset.
    """

    def __init__(self, dataset = None):
        """
        Initialize the DataLoader with the path to the dataset.

        Args:
            dataset (str): The path to the CSV dataset file.
        """
        self.dataset = dataset

    def load_data(self):
        """
        Load, preprocess, and split the dataset into training and testing data.

        Returns:
            torch.utils.data.DataLoader: DataLoader for training data.
            torch.utils.data.DataLoader: DataLoader for testing data.
        """
        # Load the dataset from a CSV file.
        dataframe = pd.read_csv(self.dataset)

        # Remove the 'Id' column as it's not needed.
        dataframe.drop(['Id'], axis = 1, inplace = True)

        # Display the column names of the dataset.
        print("Features of the dataset: {}".format(list(dataframe.columns)).capitalize(), '\n')

        # Encode the target class into numerical representation.
        dataframe = self._encode_target(dataframe = dataframe)

        # Perform data preprocessing.
        dataframe = self._preprocess_data(dataframe = dataframe)

        # Split the data into training and testing sets and create data loaders.
        train_loader, test_loader = self._create_data_loaders(dataframe = dataframe)

        return train_loader, test_loader

    def _encode_target(self, dataframe = None):
        """
        Encode the target class into numerical representation.

        Args:
            dataframe (pd.DataFrame): The dataset in DataFrame format.

        Returns:
            pd.DataFrame: The encoded dataset.
        """
        if dataframe is None:
            raise DataFrameException("DataFrame should be provided.")
        else:
            # Convert the target class into numerical representation.
            dataframe.iloc[:, -1] = dataframe.iloc[:, -1].\
                map({attribute: index for index, attribute in enumerate(dataframe.iloc[:, -1].value_counts().index)})

            return dataframe

    def _preprocess_data(self, dataframe = None):
        """
        Perform data preprocessing.

        Args:
            dataframe (pd.DataFrame): The dataset in DataFrame format.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        columns = dataframe.columns

        # Data preprocessing.
        X = dataframe.iloc[:, :-1].values
        y = dataframe.iloc[:, -1].values

        standard_scaler  = StandardScaler()
        transformed_data = standard_scaler.fit_transform(X)

        X = pd.DataFrame(transformed_data, columns = columns[:-1])
        y = pd.DataFrame(y, columns = [columns[-1]])

        dataframe = pd.concat([X, y], axis = 1)

        return dataframe

    def _create_data_loaders(self, dataframe = None):
        """
        Create data loaders for training and testing data.

        Args:
            dataframe (pd.DataFrame): The preprocessed dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader for training data.
            torch.utils.data.DataLoader: DataLoader for testing data.
        """
        # Initialize batch size.
        BATCH_SIZE = 16

        # Split the dataset into training and testing sets.
        X = dataframe.iloc[:, :-1].values
        y = dataframe.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

        y_train = y_train.astype(np.float32)
        y_test  = y_test.astype(np.float32)

        # Convert data to tensors compatible with PyTorch.
        X_train = torch.tensor(data = X_train, dtype = torch.float32)
        X_test  = torch.tensor(data = X_test,  dtype = torch.float32)

        y_train = torch.tensor(data = y_train, dtype = torch.float32)
        y_test  = torch.tensor(data = y_test,  dtype = torch.float32)

        print("X_train shape: {} ".format(X_train.shape), '\n')
        print("y_train shape: {} ".format(y_train.shape), '\n')
        print("X_test shape : {} ".format(X_test.shape), '\n')
        print("y_test shape : {} ".format(y_test.shape), '\n')
        print("_" * 50)

        # Create data loaders for training and testing data.
        train_loader = DataLoader(dataset    = list(zip(X_train, y_train)),
                                  batch_size = BATCH_SIZE,
                                  shuffle    = True)

        test_loader = DataLoader(dataset     = list(zip(X_test, y_test)),
                                 batch_size  = BATCH_SIZE,
                                 shuffle     = True)

        train_data, train_label = next(iter(train_loader))

        print("\nBatch size of train loader # {} ".format(train_loader.batch_size), '\n')
        print("Batch size of test loader    # {} ".format(test_loader.batch_size), '\n')
        print("Train data with batch size   # {} ".format(train_data.shape), '\n')
        print("Test data with batch size    # {} ".format(train_label.shape), '\n')
        print("_"*50,'\n\n')

        return train_loader, test_loader

if __name__ == "__main__":
    dataloader = dataloader(dataset = 'C:/Users/atiku/Downloads/archive (6)/Iris.csv')
    train_loader, test = dataloader.load_data()
