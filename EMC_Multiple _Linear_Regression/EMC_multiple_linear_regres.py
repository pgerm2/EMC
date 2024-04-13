# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Cond_Emis_MR.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# ****Troubleshooting Test to show why was getting "ValueError: x and y must be the same size" error
# ****Print X_test shape. What do you see? 
# ****X_train is 2d (matrix with a single column), while y_train 1d (vector). In turn you get different sizes.
# print(X_test)
# print()
# print(y_test)

# # Visualising the Test set results
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X, regressor.predict(X), color = 'blue')
# plt.title('Frequency vs Noise Level (Test set)')
# plt.xlabel('Frequency (Mhz)')
# plt.ylabel('Noise Level (dBuv)')
# plt.show()

# Evaluating the Model Performance
from sklearn.metrics import r2_score
perf = r2_score(y_test, y_pred)
print('\n' + "Performance Metric")
print(perf)