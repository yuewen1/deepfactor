from model import *
import numpy as np
import csv
import pandas as pd
import torch
import torchvision as ptv
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Net

def load_data(dataname):
	data = pd.read_csv(dataname)
	name = data.columns.values.tolist()
	data = data.dropna(axis = 1, how = 'any')
	data.to_csv("process_data.csv",index=False,sep=',')
	data = data[:1000]
	print(data.columns[:])
	print(data.columns[7:10])
	label = data[data.columns[7:10]].to_numpy()
	label = label.astype(np.float64)
	label_tensor = torch.from_numpy(label)
	label_tensor = label_tensor.type(torch.FloatTensor)
	label_tensor = label_tensor.view(-1, 3)
	feature = data.drop(data.columns[0:24], axis=1)
	feature = feature.drop(['div_yid', 'sub_ind_code_y'], axis = 1)
	feature = feature.to_numpy()
	feature = feature.astype(np.float64)
	tensor = torch.from_numpy(feature)
	feature_tensor = tensor.type(torch.FloatTensor)

	return feature_tensor, label_tensor


if __name__ == '__main__':
	dataname = 'AMZN_return_factor.csv'
	load_data(dataname)
	data, label = load_data(dataname)
	input_dim = 33
	hidden_pred = 15
	net = Net(input_dim, hidden1 = 108, hidden2 = 108, hidden_pred = 15, output_dim = 1)
	encoder = Encoder(factor_dim = 27,  hidden_dim = 10, time_step = 5, num_layer = 3)
	optimizer = torch.optim.SGD([{'params':net.parameters()},{'params':encoder.parameters()}], lr=0.2, momentum = 0.9)
	# optimizer = torch.optim.SGD(net.parameters(), lr = 0.2, momentum = 0.9)
	loss_func = torch.nn.MSELoss()
	encoder = Encoder(factor_dim = 15,  hidden_dim = 10, time_step = 5, num_layer = 3)
	# # attention = Attention(hidden_dim = 10, encoder_dim = 10) 
	epoch = 500
	label = label[:, 0].view((-1,1))
	for i in range(epoch):
		print (i)
		deep_factor = net(data) 
		pack_factor = deep_factor.view(4, 250, 15)
		label = label.view(4, 250, 1)
	# 	# prediction = encoder(pack_factor)
	# 	# num_layer, batch, hidden_dim = prediction.shape
	# 	# for i in range(2):
	# 	# 	print (i)
	# 	# 	weight = attention(out[:, i, :], hn[:, i, :], cn[:, i, :])
	# 	# pack_factor = deep_factor.permute(1, 0, 2)

	# 	loss = loss_func(prediction, label)
	# 	optimizer.zero_grad()
	# 	loss.backward() 
	# 	optimizer.step()
	# 	print (loss)


		# if i % 5 == 0:
		# 	plot(data, label, prediction)









