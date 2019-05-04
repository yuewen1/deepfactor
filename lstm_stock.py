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


def load_data(dataname, name_lst):
	data_all = pd.read_csv(dataname)
	data = data_all.iloc[0:1000]
	data = data.dropna(axis = 1, how = 'any')
	label = data.loc[:,['fut_1w_ret', 'fut_1m_ret', 'fut_2m_ret']].to_numpy()
	label = label.astype(np.float64)
	feature = data.loc[:,name_lst].to_numpy()
	feature = feature.astype(np.float64)
	return feature, label


def create_dataset(dataset, label, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(label[i + look_back - 1, :])
	return np.array(dataX), np.array(dataY)


# larger than 0 -> 1, less than 0 -> 0
def zero_or_one(x):
	h = np.zeros_like(x)
	h[x > 0] = 1
	return h


dataname = 'AMZN_return_factor.csv'
name_lst = ['prev_1m_ret', 'prev_3m_ret', 'prev_6m_ret', 'prev_9m_ret', 'prev_12m_ret', 'prev_ret12m_l1m',
			'1m_6musdvol_chg', '1m_6mvol_chg', '1mvt', '3mvt', '6mvt', '12mvt', 'beta_1y', 'beta_3y', 
			'cfo_p', 'fcf_p', 'net_cfo_p', 'ep_ltm', 'bk_p', 'sales_p', 'mcap', 'fy1_p', 'fy2_p', 'eps_p_ntm',
			'hist_eps_1yg', 'roe', 'roa', 'sales_g', 'asset_tover', 'std_eps', 'si_outshs', 'earning_risk', 
			'ir_sen_2y', 'ir_sen_10y', 'ir_spread_sen', 'ted_sen', 'crb_com_sen', 'oil_sen', 'usd_idx_sen', 
			'gross_mgn', 'us3mo', 'us2yr', 'us10yr', 'pairwise_avg_corr_index_30day', 'dxy_z', 'gold']
data, label = load_data(dataname, name_lst)
print(type(data))
print(type(label))
train_data = np.zeros((int(data.shape[0]/2), data.shape[1]))
train_label = np.zeros((int(data.shape[0]/2), 1))
test_data = np.zeros((data.shape[0] - train_data.shape[0], data.shape[1]))
test_label = np.zeros((data.shape[0] - train_label.shape[0], 1))
# set the label to future 1m return
y = label[:,0].reshape(-1,1)

###### train data first half, test data second half #####
train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
train_label, test_label = y[0:train_size,:], y[train_size:len(y),:]
train_data, test_data = data[0:train_size,:], data[train_size:len(y),:]
###### take train data and test data in sequence #####
# n = int(len(y) / 4)
# for i in range(4):
# 	k = int(i / 2)
# 	if i % 2 == 0:
# 		train_data[k*n:(k+1)*n,:] = data[i*n:(i+1)*n,:]
# 		train_label[k*n:(k+1)*n,:] = y[i*n:(i+1)*n,:]
# 	else:
# 		test_data[k*n:(k+1)*n,:] = data[i*n:(i+1)*n,:]
# 		test_label[k*n:(k+1)*n,:] = y[i*n:(i+1)*n,:]

train_data_mean = np.mean(train_data, axis=0)
train_data_std = np.std(train_data, axis=0)
train_data_norm = (train_data - train_data_mean) / (train_data_std + 1e-8)
test_data_norm = (test_data - train_data_mean) / (train_data_std + 1e-8)
train_label_mean = np.mean(train_label, axis=0)
train_label_std = np.std(train_label, axis=0)
train_label_norm = (train_label - train_label_mean) / (train_label_std + 1e-8)
test_label_norm = (test_label - train_label_mean) / (train_label_std + 1e-8)

look_back = 36
trainX, trainY = create_dataset(train_data_norm, train_label_norm, look_back)
testX, testY = create_dataset(test_data_norm, test_label_norm, look_back)

# create and fit the LSTM network, optimizer=adam, 100 neurons, dropout 0.2
model = Sequential()
lstm1 = LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2]))
model.add(lstm1)
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, validation_split=0.1, epochs=50, batch_size=32, verbose=1)
# make predictions
train_predict = model.predict(trainX)
print(train_predict.shape)
train_predict = train_predict * (train_label_std + 1e-8) + train_label_mean
test_predict = model.predict(testX)
test_predict = test_predict * (train_label_std + 1e-8) + train_label_mean

# find the top ten important factors
weights = lstm1.get_weights()[0]
weights_sum = np.sum(np.abs(weights), axis=1)
importance = train_data_std * weights_sum
lst = []
for i in range(len(importance)):
	lst.append((importance[i], name_lst[i]))
sorted_lst = sorted(lst, key=lambda lst : lst[0], reverse=True)
top_ten_label = []
top_ten_value = []
y_axis = []
top_ten = sorted_lst[0:10]
for i in range(10):
	y_axis.append(i+1)
	top_ten_value.append(top_ten[i][0])
	top_ten_label.append(top_ten[i][1])

print(top_ten_label)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, test_predict))
print('Test Score: %.2f RMSE' % (testScore))

# calculate accuracy for classifying rise or fall
train_label_sign = zero_or_one(trainY)
train_predict_sign = zero_or_one(train_predict)
test_label_sign = zero_or_one(testY)
test_predict_sign = zero_or_one(test_predict)
train_acc = float((train_predict_sign == train_label_sign).astype(int).sum())  / float(train_predict.shape[0])
print('Train Accuracy: %.2f ' % (train_acc))
test_acc = float((test_label_sign == test_predict_sign).astype(int).sum())  / float(test_predict.shape[0])
print('Test Accuracy: %.2f ' % (test_acc))


plt.figure(1)
trainPredictPlot = np.empty_like(y)
trainPredictPlot[:, :] = np.nan
testPredictPlot = np.empty_like(y)
testPredictPlot[:, :] = np.nan
###### train data first half, test data second half #####
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
testPredictPlot[len(train_predict)+(look_back*2)+1:len(y)-1, :] = test_predict
###### take train data and test data in sequence #####
# for i in range(20):
# 	k = int(i / 2)
# 	if i % 2 == 0:
# 		trainPredictPlot[i*48:(i+1)*48,:] = train_predict[k*48:(k+1)*48,:]
# 	else:
# 		testPredictPlot[i*48:(i+1)*48,:] = test_predict[k*48:(k+1)*48,:]
plt.figure(1)
l1, = plt.plot(y, label="original return")
l2, = plt.plot(trainPredictPlot, label="train return prediction")
l3, = plt.plot(testPredictPlot, label="test return prediction")
plt.xlabel("time")
# if we use y = label[:,1].reshape(-1,1), it will be "future 1 month return"
plt.ylabel("future 1 week return")
plt.title("future 1 week return vs time")
plt.legend(handles=[l1, l2, l3])
plt.show()

plt.figure(2)
plt.xscale('log')
plt.barh(y_axis, top_ten_value, alpha=0.5)
plt.yticks(y_axis, top_ten_label)
plt.ylabel("factor name")
plt.xlabel("factor importance")
plt.title("top 10 important factors for LSTM model")
plt.show()
