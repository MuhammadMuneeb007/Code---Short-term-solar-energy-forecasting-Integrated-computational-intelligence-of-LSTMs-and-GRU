from math import sqrt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import sklearn
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from regressormetricgraphplot import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, median_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, HuberRegressor, \
LassoLars, BayesianRidge, TweedieRegressor, OrthogonalMatchingPursuitCV ,PoissonRegressor , GammaRegressor, ARDRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge 
from bokeh.io import export_png, export_svgs
from bokeh.models import ColumnDataSource, DataTable, TableColumn
import dataframe_image as dfi
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,StackingRegressor, VotingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor , RadiusNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
from math import sqrt
from tabnanny import verbose
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import median_absolute_error
from tensorflow.keras.optimizers import RMSprop, SGD, Adam , Adadelta
from pandas import concat
from numpy import concatenate
from pandas import DataFrame
from math import e 
import math


from sklearn.preprocessing import MinMaxScaler

results = pd.DataFrame(columns = ['MSE','MAE','RMSE','R2Score','Explainedvariancescore'])

def Stackedgrulstmbilstm(model1,model2,X_train, X_test, y_train, y_test):
  model = keras.models.Sequential()
  if model1=="LSTM":  
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.5))
  elif model1=="GRU":
    model.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(GRU(64, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.5))
  elif model1=="BILSTM":
    model.add(Bidirectional(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.5))

  if model2=="LSTM":  
    model.add(LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add((LSTM(16,return_sequences=False)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
  elif model2=="GRU":
    model.add(GRU(32, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add((GRU(16,return_sequences=False)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
  elif model2=="BILSTM":
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(Bidirectional((LSTM(16,return_sequences=False))))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))


 
  optimizer = keras.optimizers.Adam()
  model.compile(loss='mean_squared_error', optimizer=optimizer)

  history2= model.fit(X_train, y_train,epochs=30,batch_size=20,validation_split=0.1,shuffle=False,verbose=1)



  y_pred = model.predict(X_test)
  tmse = mean_squared_error(y_pred,y_test)
  tmae = mean_absolute_error(y_pred,y_test)
  trmse = sqrt(mean_squared_error(y_pred,y_test))
  tr2_Score = r2_score(y_pred,y_test)

  texplained_variance_Score = explained_variance_score(y_pred,y_test)
  results.loc[model1+"_"+model2] = [tmse,tmae,trmse,tr2_Score,texplained_variance_Score]

def lstmmodel(X_train, X_test, y_train, y_test):
  model = keras.models.Sequential()
  model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(LSTM(64, return_sequences=True))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(LSTM(32, return_sequences=True))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add((LSTM(16,return_sequences=False)))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(1)) 
  optimizer = keras.optimizers.Adam()
  model.compile(loss='mean_squared_error', optimizer=optimizer)

  history2= model.fit(X_train, y_train,epochs=30,batch_size=20,validation_split=0.1,shuffle=False,verbose=1)



  y_pred = model.predict(X_test)
  tmse = mean_squared_error(y_pred,y_test)
  tmae = mean_absolute_error(y_pred,y_test)
  trmse = sqrt(mean_squared_error(y_pred,y_test))
  tr2_Score = r2_score(y_pred,y_test)
  texplained_variance_Score = explained_variance_score(y_pred,y_test)
  results.loc['LSTM'] = [tmse,tmae,trmse,tr2_Score,texplained_variance_Score]

def grumodel(X_train, X_test, y_train, y_test):
  model = keras.models.Sequential()
  model.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(GRU(64, return_sequences=True))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(GRU(32, return_sequences=True))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add((GRU(16,return_sequences=False)))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(1))
  optimizer = keras.optimizers.Adam()
  model.compile(loss='mean_squared_error', optimizer=optimizer)

  history2= model.fit(X_train, y_train,epochs=30,batch_size=20,validation_split=0.1,shuffle=False,verbose=1)



  y_pred = model.predict(X_test)
  tmse = mean_squared_error(y_pred,y_test)
  tmae = mean_absolute_error(y_pred,y_test)
  trmse = sqrt(mean_squared_error(y_pred,y_test))
  tr2_Score = r2_score(y_pred,y_test)
  texplained_variance_Score = explained_variance_score(y_pred,y_test)
  results.loc['GRU'] = [tmse,tmae,trmse,tr2_Score,texplained_variance_Score]

def bilstmmodel(X_train, X_test, y_train, y_test):
  model = keras.models.Sequential()
  model.add(Bidirectional(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True)))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(Bidirectional(LSTM(64, return_sequences=True)))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(Bidirectional(LSTM(32, return_sequences=True)))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(Bidirectional(LSTM(16,return_sequences=False)))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(1)) 
  optimizer = keras.optimizers.Adam()
  model.compile(loss='mean_squared_error', optimizer=optimizer)

  history2= model.fit(X_train, y_train,epochs=30,batch_size=20,validation_split=0.1,shuffle=False,verbose=1)



  y_pred = model.predict(X_test)
  tmse = mean_squared_error(y_pred,y_test)
  tmae = mean_absolute_error(y_pred,y_test)
  trmse = sqrt(mean_squared_error(y_pred,y_test))
  tr2_Score = r2_score(y_pred,y_test)
  texplained_variance_Score = explained_variance_score(y_pred,y_test)
  results.loc['BILSTM'] = [tmse,tmae,trmse,tr2_Score,texplained_variance_Score]

df = pd.read_csv('weather.csv')
df.drop(['Timestamp',' UV '],axis=1,inplace=True)

df=df.dropna()

 

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))
 

f_columns = ['Temp', 'Chill', 'Humid', 'Dewpt',' Wind ','HiWind','Rain ','Barom',' ET  ']
f_transformer = MinMaxScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)
test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)
 
solar_transformer = MinMaxScaler()
solar = solar_transformer.fit(train[['Solar']])
train['Solar'] = solar.transform(train[['Solar']])
test['Solar'] = solar.transform(test[['Solar']])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Specify the timestep.
t = [1,3,6,12,24]

for time_steps in t:
 X_train, y_train = create_dataset(train, train.Solar, time_steps)
 X_test, y_test = create_dataset(test, test.Solar, time_steps)

 y_train = y_train.reshape(-1,1)
 y_test = y_test.reshape(-1,1)
 
 grumodel(X_train, X_test, y_train, y_test)
 lstmmodel(X_train, X_test, y_train, y_test)
 bilstmmodel(X_train, X_test, y_train, y_test)
 Stackedgrulstmbilstm("LSTM","GRU",X_train, X_test, y_train, y_test)
 Stackedgrulstmbilstm("LSTM","BILSTM",X_train, X_test, y_train, y_test)
 Stackedgrulstmbilstm("BILSTM","LSTM",X_train, X_test, y_train, y_test)
 Stackedgrulstmbilstm("BILSTM","GRU",X_train, X_test, y_train, y_test)
 Stackedgrulstmbilstm("GRU","LSTM",X_train, X_test, y_train, y_test)
 Stackedgrulstmbilstm("GRU","BILSTM",X_train, X_test, y_train, y_test)  

 print("Results for timestep:",str(t))
 print(results.to_markdown()) 
 results = pd.DataFrame(columns = ['MSE','MAE','RMSE','R2Score','Explainedvariancescore'])

