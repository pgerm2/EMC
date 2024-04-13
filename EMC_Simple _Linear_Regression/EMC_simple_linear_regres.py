# EMC Cond Emis Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Cond_Emis.csv')
X = dataset.iloc[:, :-1].values #selects all col except last one
y = dataset.iloc[:, -1].values # selects automaticlly the depend var vector because it selects the last col

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Making the Confusion Matrix. CANNOT USE IT. CAN ONLY BE USED IN CLASSIFICATION.

# # Visualising the Training set results 
# plt.scatter(X_train, y_train, color = 'red') #code for drawing red dots
# plt.plot(X_train, regressor.predict(X_train), color = 'blue') #code for drawing blue regression line
# plt.title('Frequency vs Noise Level (Training set)')
# plt.xlabel('Frequency (Mhz)')
# plt.ylabel('Noise Level (dBuv)')
# plt.show()

# # Visualising the Training set results Ex 
# ax = plt.scatter(X_train, y_train, color = 'red') #code for drawing red dots
# rng = np.arange(10)
# db = 78 + rng
plt.scatter(X_train, y_train, color = 'red') #code for drawing red dots
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #code for drawing blue regression line
plt.title('Frequency vs Noise Level (Training set)')
plt.xlabel('Frequency (Mhz)')
plt.ylabel('Noise Level (dBuv)')
#ax.axes.set_xlim(xmin=db[0], xmax=db[-1])
# ax.axes.set_ylim(ymin=db[0], ymax=db[-1])
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Frequency vs Noise Level (Test set)')
plt.xlabel('Frequency (Mhz)')
plt.ylabel('Noise Level (dBuv)')
plt.show()

# Evaluating the Model Performance
from sklearn.metrics import r2_score
perf = r2_score(y_test, y_pred)
print(perf)

##*****************************************************************

# import matplotlib.pyplot as plt
# import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (10 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

##******************************************************************

# import matplotlib.pyplot as plt
# import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# # plt.subplot(111)
# # plt.imshow(np.random.random((10, 10)))
# plt.subplot(111) #size of
# plt.imshow(np.random.random((5, 5))) #number of sq inside graph

# #Bar plot
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes((0.85, 0.1, 0.075, 0.8))
# plt.colorbar(cax=cax)

# plt.show()