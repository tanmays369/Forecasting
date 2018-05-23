import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import random
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from matplotlib.pylab import rcParams

df= pd.read_csv('export.csv', delimiter= '\t', names= ['index', 'lane_id', 'Count', 'Time', 'Timestamp', 'Green'], parse_dates= ['Timestamp'], index_col= 'Timestamp')

for i in df.index:
    i= i.time()

df.drop(['index'], axis= 1, inplace= True)

sample_cols= ['Green', 'lane_id']
for i in sample_cols:
    df[i].astype('category')
#df= pd.get_dummies(data=df, columns=['lane_id', 'Green'])
df= pd.get_dummies(data=df, columns=['Green'])
df.drop(['Green_0'], 1, inplace= True)

df_lane1= df[df['lane_id']== 1]
df_lane2= df[df['lane_id']== 2]
df_lane3= df[df['lane_id']== 3]
df_lane4= df[df['lane_id']== 4]

rcParams['figure.figsize'] = 15, 6

'''
print('--------------df------------------')
print(df.dtypes)
'''

def ts_plot_lame1(dataframe):
    ts1 = dataframe['Count']
    #print(ts1.tail())
    '''
    print(ts.head(10))
    
    print(df.index)
    '''
    #print(ts1[:'2018-03-30 10:03:59'])
    #print(ts1['2018-03-30 09'])
    plt.plot(ts1)
    plt.show()

dataframe= df_lane1
ts_plot_lame1(dataframe)
dataframe= df_lane2
ts_plot_lame1(dataframe)
dataframe= df_lane3
ts_plot_lame1(dataframe)
dataframe= df_lane4
ts_plot_lame1(dataframe)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original', alpha= 0.8)
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std', alpha= 0.8)
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def ret_ts_lame(dataframe):
    ts= dataframe['Count']
    test_stationarity(ts)
ret_ts_lame(df_lane1)
ret_ts_lame(df_lane2)
ret_ts_lame(df_lane3)
ret_ts_lame(df_lane4)


def ret_ts(dataframe):
    return dataframe.Count

def ret_log_transform(dataframe):
    ts= ret_ts(dataframe)
    ts_log = np.log(ts)
    return ts_log
    
def plot_ret_log_transform(dataframe):
    ob= ret_log_transform(dataframe)
    plt.plot(ob)
    plt.show()
    
plot_ret_log_transform(df_lane1)
plot_ret_log_transform(df_lane2)
plot_ret_log_transform(df_lane3)
plot_ret_log_transform(df_lane4)

ret_log_transform(df_lane1)
ret_log_transform(df_lane2)
ret_log_transform(df_lane3)
ret_log_transform(df_lane4)

def ret_moving_avg(dataframe):
    moving_avg = pd.rolling_mean(ret_log_transform(dataframe),5)
    plt.plot(ret_log_transform(dataframe))
    plt.plot(moving_avg, color='red')
    plt.show()
    return moving_avg

mov_avg1= ret_moving_avg(df_lane1)
mov_avg2= ret_moving_avg(df_lane2)
mov_avg3= ret_moving_avg(df_lane3)
mov_avg4= ret_moving_avg(df_lane4)

from statsmodels.tsa.seasonal import seasonal_decompose

def see_diff(dataframe):
    ts_log_diff = ret_log_transform(dataframe) - ret_log_transform(dataframe).shift()
    plt.plot(ts_log_diff)
    plt.show()
    ts_log_diff.dropna(inplace=True)
    return ts_log_diff
    #test_stationarity(ts_log_diff)
    '''
    decomposition = seasonal_decompose(ret_log_transform(dataframe))

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ret_log_transform(dataframe), label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    '''
#see_diff(df_lane1)
#see_diff(df_lane2)
#see_diff(df_lane3)
#see_diff(df_lane4)

from statsmodels.tsa.arima_model import ARIMA

def draw_ARIMA(dataframe):
    model = ARIMA(ret_log_transform(dataframe), order=(2, 1, 0))  
    results_AR = model.fit(disp=-1)
    plt.figure(figsize= (10, 5))
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-see_diff(dataframe))**2))
    #plt.plot(see_diff(dataframe))
    plt.plot(results_AR.fittedvalues, color='red')
    plt.show()

draw_ARIMA(df_lane1)
draw_ARIMA(df_lane2)
draw_ARIMA(df_lane3)
draw_ARIMA(df_lane4)
