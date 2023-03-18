import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# reading the data
data = pd.read_csv('/content/clean_weather.csv', index_col = 0)
data.ffill()

data.plot.scatter('tmax', 'tmax_tomorrow')

data.corr() # to see what correlation coeficient is and it is 0.818820

data.plot.scatter('tmax', 'tmax_tomorrow')
prediction = lambda x, w1 = .82, b = 11.99: x * w1 + b
plt.plot([30,120], [prediction(30), prediction(120)], "red") # this red line is the best linear prediction to do to make the prediction of tmax_tomorrow



def mse(actual, predicted):
  return np.mean((actual - predicted) ** 2)

print(mse(data["tmax_tomorrow"], prediction(data["tmax"])))