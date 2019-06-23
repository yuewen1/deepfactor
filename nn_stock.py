import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
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
train_data = np.zeros((int(data.shape[0]/2), data.shape[1]))
train_label = np.zeros((int(data.shape[0]/2), 1))
test_data = np.zeros((data.shape[0] - train_data.shape[0], data.shape[1]))
test_label = np.zeros((data.shape[0] - train_label.shape[0], 1))
# set the label to future 1w return
y = label[:,0].reshape(-1,1)

###### train data first half, test data second half #####
train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
train_label, test_label = y[0:train_size,:], y[train_size:len(y),:]
train_data, test_data = data[0:train_size,:], data[train_size:len(y),:]

###### take train data and test data in sequence #####
# for i in range(20):
# 	k = int(i / 2)
# 	if i % 2 == 0:
# 		train_data[k*48:(k+1)*48,:] = data[i*48:(i+1)*48,:]
# 		train_label[k*48:(k+1)*48,:] = y[i*48:(i+1)*48,:]
# 	else:
# 		test_data[k*48:(k+1)*48,:] = data[i*48:(i+1)*48,:]
# 		test_label[k*48:(k+1)*48,:] = y[i*48:(i+1)*48,:]

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)
train_data_mean = np.mean(train_data, axis=0)
train_data_std = np.std(train_data, axis=0)
train_data_norm = (train_data - train_data_mean) / (train_data_std + 1e-8)
test_data_norm = (test_data - train_data_mean) / (train_data_std + 1e-8)
train_label_mean = np.mean(train_label, axis=0)
train_label_std = np.std(train_label, axis=0)
train_label_norm = (train_label - train_label_mean) / (train_label_std + 1e-8)
test_label_norm = (test_label - train_label_mean) / (train_label_std + 1e-8)


model = Sequential()
dense1 = Dense(196, input_shape=(46,), activation='tanh')
model.add(dense1)
model.add(Dense(98, activation='tanh'))
model.add(Dense(46, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(train_data_norm, train_label_norm, validation_split=0.1, epochs=50, batch_size=8,verbose=1)
train_predict = model.predict(train_data_norm)
train_predict = train_predict * (train_label_std + 1e-8) + train_label_mean
test_predict = model.predict(test_data_norm)
test_predict = test_predict * (train_label_std + 1e-8) + train_label_mean

# find the top ten important factors
weights = dense1.get_weights()[0]
print(weights.shape)
weights_sum = np.sum(np.abs(weights), axis=1)
importance = train_data_std * weights_sum
# print(importance)
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

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_label_norm, train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_label_norm, test_predict))
print('Test Score: %.2f RMSE' % (testScore))
# calculate accuracy for classifying rise or fall
train_label_sign = zero_or_one(train_label_norm)
train_predict_sign = zero_or_one(train_predict)
test_label_sign = zero_or_one(test_label_norm)
test_predict_sign = zero_or_one(test_predict)
train_acc = float((train_predict_sign == train_label_sign).astype(int).sum())  / float(train_predict.shape[0])
print('Train Accuracy: %.2f ' % (train_acc))
test_acc = float((test_label_sign == test_predict_sign).astype(int).sum())  / float(test_predict.shape[0])
print('Test Accuracy: %.2f ' % (test_acc))

print(train_predict.shape)

plt.figure(1)
trainPredictPlot = np.empty_like(y)
testPredictPlot = np.empty_like(y)
trainPredictPlot[:, :] = np.nan
testPredictPlot[:, :] = np.nan

###### train data first half, test data second half #####
trainPredictPlot[0:len(train_predict), :] = train_predict
testPredictPlot[len(train_predict):len(y), :] = test_predict

###### take train data and test data in sequence #####
# for i in range(20):
# 	k = int(i / 2)
# 	if i % 2 == 0:
# 		trainPredictPlot[i*48:(i+1)*48,:] = train_predict[k*48:(k+1)*48,:]
# 	else:
# 		testPredictPlot[i*48:(i+1)*48,:] = test_predict[k*48:(k+1)*48,:]

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
plt.title("top 10 important factors for neural network model")
plt.show()
