#####################################################################################################################
#   Linear Regression using Gradient Descent
#
#   train - training dataset - It is in the same folder where this file is
#         - you can assume the last column will the label column
#   test - test dataset - It is in the same folder where this file is
#         - you can assume the last column will the label column
#
#   PreProcessing of all the attributes except for output attribute is done.
#
#####################################################################################################################

import numpy as np
import pandas as pd

#  * @author vagdevi
#  * @author anki

class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        self.df = pd.read_csv(train)
        # calling preProcess before geting the data to X. Y is not effected as it is taken care in the method.
        self.preProcess()
        self.df.insert(0, 'X0', 1)
        self.nrows, self.ncols = self.df.shape[0], self.df.shape[1]
        self.X =  self.df.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = self.df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

    #   Pre-processing the dataset
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function
    def preProcess(self):
        self.df = LinearRegression.preProcessCsv(self.df)

    @staticmethod
    def preProcessCsv(df):
        # removing null values in the data. This data has no null values.
        df = df.dropna()
        # There are no categorical variables
        # Removing duplicate values from the data
        df = df.drop_duplicates()

        # Scaling the features between 0 and 1.
        for column in df.columns[1:10]:
            df[column] = df[column] - df[column].min()
            df[column] = df[column] / df[column].max()
        return df

    # Below is the training function
    def train(self, epochs = 1000, learning_rate = 0.01):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X, self.W)
            # Find error
            error = h - self.y
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X.T, error)

        return self.W, error

    # predict on test dataset
    def predict(self, file):
        testDF = pd.read_csv(file)
        testDF = LinearRegression.preProcessCsv(testDF)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = 1/(2.0*nrows) * np.dot(error.T, error)
        return mse


if __name__ == "__main__":
    model = LinearRegression("train.csv")
    W, e = model.train()
    mse = model.predict("train.csv")
    print("Mean sqaure error for train data::%.2f" %mse)
    mse = model.predict("test.csv")
    print("Mean sqaure error for test data:%.2f" %mse)
