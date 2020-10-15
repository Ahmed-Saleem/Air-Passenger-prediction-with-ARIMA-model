import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


data= pd.read_csv('AirPassengers.csv')
data['Month']=pd.to_datetime(data['Month'], infer_datetime_format=True)
indexed_data= data.set_index(['Month'])
plt.plot(indexed_data)
plt.xlabel('Time')
plt.ylabel('No. of Passengers')

#Rolling statistics test
roll_mean= indexed_data.rolling(window=12).mean()
roll_std= indexed_data.rolling(window=12).std()
plt.plot(indexed_data, color='blue', label='orignal')
plt.plot(roll_mean, color='red', label='mean')
plt.plot(roll_std, color='black', label='standard deviation')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
plt.show()

#Dickey-Fuller test
dftest=adfuller(indexed_data['#Passengers'], autolag='AIC')
dfoutput=pd.DataFrame(data=dftest[0:4], index=['Test statistic', 'p-value', '#Lags used', '#Observations used'])
dfcritical_values= pd.DataFrame(data=dftest[4].items())

#logscale
indexed_data_logscale= np.log(indexed_data)
plt.plot(indexed_data_logscale)

#difference between logscale values and rolling means
roll_mean_log= indexed_data_logscale.rolling(window=12).mean()
roll_std_log= indexed_data_logscale.rolling(window=12).std()
plt.plot(indexed_data_logscale, color='blue', label='orignal')
plt.plot(roll_mean_log, color='red', label='mean')
plt.plot(roll_std_log, color='black', label='standard deviation')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
plt.show()
indexed_data_logscale_minus_roll_mean= indexed_data_logscale-roll_mean_log
indexed_data_logscale_minus_roll_mean.dropna(inplace=True)

#Rolling statistics test for difference between logscale values and rolling means
roll_mean= indexed_data_logscale_minus_roll_mean.rolling(window=12).mean()
roll_std= indexed_data_logscale_minus_roll_mean.rolling(window=12).std()
plt.plot(indexed_data_logscale_minus_roll_mean, color='blue', label='orignal')
plt.plot(roll_mean, color='red', label='mean')
plt.plot(roll_std, color='black', label='standard deviation')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
plt.show()

#Dickey-Fuller test for difference between logscale values and rolling means
dftest=adfuller(indexed_data_logscale_minus_roll_mean['#Passengers'], autolag='AIC')
dfoutput=pd.DataFrame(data=dftest[0:4], index=['Test statistic', 'p-value', '#Lags used', '#Observations used'])
dfcritical_values= pd.DataFrame(data=dftest[4].items())

#difference between logscale values and weighted means
ExponentialDecayWeightedAverage= indexed_data_logscale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexed_data_logscale, label='orignal', color='blue')
plt.plot(ExponentialDecayWeightedAverage, label= 'weighted mean', color='red')
indexed_data_logscale_minus_weighted_mean= indexed_data_logscale-ExponentialDecayWeightedAverage

#Rolling statistics test for difference between logscale values and weighted means
roll_mean= indexed_data_logscale_minus_weighted_mean.rolling(window=12).mean()
roll_std= indexed_data_logscale_minus_weighted_mean.rolling(window=12).std()
plt.plot(indexed_data_logscale_minus_weighted_mean, color='blue', label='orignal')
plt.plot(roll_mean, color='red', label='mean')
plt.plot(roll_std, color='black', label='standard deviation')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
plt.show()

#Dickey-Fuller test for difference between logscale values and weighted means
dftest=adfuller(indexed_data_logscale_minus_weighted_mean['#Passengers'], autolag='AIC')
dfoutput=pd.DataFrame(data=dftest[0:4], index=['Test statistic', 'p-value', '#Lags used', '#Observations used'])
dfcritical_values= pd.DataFrame(data=dftest[4].items())

#determining the value of d
data_logDshifting= indexed_data_logscale-indexed_data_logscale.shift()
data_logDshifting.dropna(inplace=True)

#Rolling statistics test for shifted data
roll_mean= data_logDshifting.rolling(window=12).mean()
roll_std= data_logDshifting.rolling(window=12).std()
plt.plot(data_logDshifting, color='blue', label='orignal')
plt.plot(roll_mean, color='red', label='mean')
plt.plot(roll_std, color='black', label='standard deviation')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
plt.show()

#Dickey-Fuller test for shifted data
dftest=adfuller(data_logDshifting['#Passengers'], autolag='AIC')
dfoutput=pd.DataFrame(data=dftest[0:4], index=['Test statistic', 'p-value', '#Lags used', '#Observations used'])
dfcritical_values= pd.DataFrame(data=dftest[4].items())

decomposition= seasonal_decompose(indexed_data_logscale)
trend=decomposition.trend
seasonal=decomposition.seasonal
residuals=decomposition.resid
plt.subplot(411)
plt.plot(indexed_data_logscale, label='orignal')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residuals, label='residuals')
plt.legend(loc='best')

#determining values of p and q for ACF and PACF graphs
lag_acf= acf(data_logDshifting, nlags=20)
lag_pacf= pacf(data_logDshifting, nlags=20)
#acf plot
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_logDshifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_logDshifting)),linestyle='--',color='gray')
plt.title('Autocorelation function')
#pacf plot
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_logDshifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_logDshifting)),linestyle='--',color='gray')
plt.title('Partial autocorelation function')
plt.tight_layout()

#ARIMA model
model=ARIMA(indexed_data_logscale, order=(2, 1, 2))
results_AR= model.fit(disp=-1)
plt.plot(data_logDshifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-data_logDshifting['#Passengers'])**2))

#Prediction plot for next ten years
results_AR.plot_predict(1,264)

#Prediction data for next ten years
results_AR.forecast(steps=120)








