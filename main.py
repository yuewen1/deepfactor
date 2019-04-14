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

def load_data(dataname, labelname):
	data = pd.read_csv(dataname)
	label = pd.read_csv(dataname)
	print (data.shape)
	data = data.fillna(0)
	label = data.fillna(0)
	data = data.values
	label = label.values
	data1 = data[0:1000]
	label1 = label[0:1000]
	label1 = label1[:, 0]
	data1 = np.array(data1, dtype = np.float64)
	label1 = np.array(label1, dtype = np.float64)
	tensor = torch.from_numpy(data1)
	label_tensor = torch.from_numpy(label1)
	tensor = tensor.type(torch.FloatTensor)
	label_tensor = label_tensor.type(torch.FloatTensor)
	label_tensor = label_tensor.view(-1, 1)
	return tensor, label_tensor
if __name__ == '__main__':
	dataname = 'data.csv'
	labelname = 'label.csv'
	data, label = load_data(dataname, labelname)
	net = Net(input_dim = 54, hidden1 = 108, hidden2 = 108, hidden_pred = 27, output_dim = 1)
	encoder = Encoder(factor_dim = 27,  hidden_dim = 10, time_step = 5, num_layer = 3)
	# optimizer = torch.optim.SGD([{'params':net.parameters()},{'params':encoder.parameters()}], lr=0.2, momentum = 0.9)
	optimizer = torch.optim.SGD(net.parameters(), lr = 0.2, momentum = 0.9)
	loss_func = torch.nn.MSELoss()
	encoder = Encoder(factor_dim = 27,  hidden_dim = 10, time_step = 5, num_layer = 3)
	# attention = Attention(hidden_dim = 10, encoder_dim = 10) 
	epoch = 500

	for i in range(epoch):
		print (i)
		prediction = net(data)
		# deep_factor = net(data)  
		# pack_factor = deep_factor.view(4, 250, 27)
		# label = label.view(4, 250, 1)
		# prediction = encoder(pack_factor)
		# num_layer, batch, hidden_dim = prediction.shape
		# for i in range(2):
		# 	print (i)
		# 	weight = attention(out[:, i, :], hn[:, i, :], cn[:, i, :])
		# pack_factor = deep_factor.permute(1, 0, 2)

		loss = loss_func(prediction, label)
		optimizer.zero_grad()
		loss.backward() 
		optimizer.step()
		print (loss)


		# if i % 5 == 0:
		# 	plot(data, label, prediction)









