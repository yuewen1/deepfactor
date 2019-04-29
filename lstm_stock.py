import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time #helper libraries
from keras import regularizers


def load_data(dataname):
	data = pd.read_csv(dataname)
	train_data = data.iloc[0:960]
	train_data = train_data.dropna(axis = 1, how = 'any')
	label = train_data.loc[:,['fut_1w_ret', 'fut_1m_ret', 'fut_2m_ret']].to_numpy()
	label = label.astype(np.float64)
	feature = train_data.loc[:,['prev_1m_ret', 'prev_3m_ret', 'prev_6m_ret', 'prev_9m_ret', 'prev_12m_ret', 'prev_ret12m_l1m',
	 							'1m_6musdvol_chg', '1m_6mvol_chg', '1mvt', '3mvt', '6mvt', '12mvt', 'beta_1y', 'beta_3y', 
	 							'cfo_p', 'fcf_p', 'net_cfo_p', 'ep_ltm', 'bk_p', 'sales_p', 'mcap', 'fy1_p', 'fy2_p', 'eps_p_ntm',
	 							'hist_eps_1yg', 'roe', 'roa', 'sales_g', 'asset_tover', 'std_eps', 'si_outshs', 'earning_risk', 
	 							'ir_sen_2y', 'ir_sen_10y', 'ir_spread_sen', 'ted_sen', 'crb_com_sen', 'oil_sen', 'usd_idx_sen', 
	 							'gross_mgn', 'us3mo', 'us2yr', 'us10yr', 'pairwise_avg_corr_index_30day', 'dxy_z', 'gold']].to_numpy()
	feature = feature.astype(np.float64)
	return feature, label

def create_dataset(dataset, label, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(label[i + look_back - 1, :])
	return np.array(dataX), np.array(dataY)

dataname = 'AMZN_return_factor.csv'
data, label = load_data(dataname)
# set the label to future 1m return
y = label[:,1].reshape(-1,1)
train_size = int(len(y) * 0.5)
test_size = len(y) - train_size
train_label, test_label = y[0:train_size,:], y[train_size:len(y),:]
train_data, test_data = data[0:train_size,:], data[train_size:len(y),:]
train_data_mean = np.mean(train_data, axis=0)
train_data_std = np.std(train_data, axis=0)
train_data_norm = (train_data - train_data_mean) / (train_data_std + 1e-8)
test_data_norm = (test_data - train_data_mean) / (train_data_std + 1e-8)
train_label_mean = np.mean(train_label, axis=0)
train_label_std = np.std(train_label, axis=0)
train_label_norm = (train_label - train_label_mean) / (train_label_std + 1e-8)
test_label_norm = (test_label - train_label_mean) / (train_label_std + 1e-8)

look_back = 128
trainX, trainY = create_dataset(train_data_norm, train_label_norm, look_back)
testX, testY = create_dataset(test_data_norm, test_label_norm, look_back)

# reshape input to be [samples, features, time steps]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

# create and fit the LSTM network, optimizer=adam, 100 neurons, dropout 0.1
model = Sequential()
# model.add(Dense(108, input_shape=(46, look_back), activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dense(108, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
model.add(LSTM(100, input_shape=(46, look_back)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, validation_split=0.1, epochs=200, batch_size=128, shuffle=True,verbose=1)
# make predictions
trainPredict = model.predict(trainX)
trainPredict = trainPredict * (train_label_std + 1e-8) + train_label_mean
# print(trainPredict)
testPredict = model.predict(testX)
testPredict = testPredict * (train_label_std + 1e-8) + train_label_mean


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(y)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = np.empty_like(y)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(y)-1, :] = testPredict
plt.plot(y)
plt.plot(trainPredictPlot)
print('testPrices:')
testPrices=y[test_size+look_back:]
print(testPrices)
print('testPredictions:')
print(testPredict)
plt.plot(testPredictPlot)
plt.show()
