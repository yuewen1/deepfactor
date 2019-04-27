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
	train_data = data.iloc[0:836]
	train_data = train_data.dropna(axis = 1, how = 'any')
	label = train_data.loc[:,['fut_1w_ret', 'fut_1m_ret', 'fut_2m_ret']].to_numpy()
	label = label.astype(np.float64)
	label_tensor = torch.from_numpy(label)
	label_tensor = label_tensor.type(torch.FloatTensor)
	label_tensor = label_tensor.view(-1, 3)
	feature = train_data.loc[:,['prev_1m_ret', 'prev_3m_ret', 'prev_6m_ret', 'prev_9m_ret', 'prev_12m_ret', 'prev_ret12m_l1m',
	 							'1m_6musdvol_chg', '1m_6mvol_chg', '1mvt', '3mvt', '6mvt', '12mvt', 'beta_1y', 'beta_3y', 
	 							'cfo_p', 'fcf_p', 'net_cfo_p', 'ep_ltm', 'bk_p', 'sales_p', 'mcap', 'fy1_p', 'fy2_p', 'eps_p_ntm',
	 							'hist_eps_1yg', 'roe', 'roa', 'sales_g', 'asset_tover', 'std_eps', 'si_outshs', 'earning_risk', 
	 							'ir_sen_2y', 'ir_sen_10y', 'ir_spread_sen', 'ted_sen', 'crb_com_sen', 'oil_sen', 'usd_idx_sen', 
	 							'gross_mgn', 'us3mo', 'us2yr', 'us10yr', 'pairwise_avg_corr_index_30day', 'dxy_z', 'gold']].to_numpy()
	feature = feature.astype(np.float64)
	tensor = torch.from_numpy(feature)
	feature_tensor = tensor.type(torch.FloatTensor)

	return feature_tensor, label_tensor


if __name__ == '__main__':
	dataname = 'AMZN_return_factor.csv'
	load_data(dataname)
	data, label = load_data(dataname)
	input_dim = 46
	hidden_pred = 23
	factor_dim = hidden_pred
	net = Net(input_dim, hidden1 = 108, hidden2 = 108, hidden_pred = 23, output_dim = 1)
	encoder = Encoder(factor_dim = 23,  hidden_dim = 10, time_step = 5, num_layer = 3)
	optimizer = torch.optim.SGD([{'params':net.parameters()},{'params':encoder.parameters()}], lr=0.01, momentum = 0.9)
	# optimizer = torch.optim.SGD(net.parameters(), lr = 0.2, momentum = 0.9)
	loss_func = torch.nn.MSELoss()
	# # attention = Attention(hidden_dim = 10, encoder_dim = 10) 
	epoch = 5
	label = label[:, 0].view((-1,1))
	seq_size = 4
	batch_size = int(len(data) / seq_size)
	index = [i - 1 for i in range(len(data) + 1) if i % seq_size == 0]
	index = index[1:]
	label = label[index]
	print (index)
	for i in range(epoch):
		print (i)
		deep_factor = net(data) 
		pack_factor = deep_factor.view(seq_size, batch_size, factor_dim)
		# label = label.view(seq_size, batch_size, 1)
		prediction = encoder(pack_factor)




		loss = loss_func(prediction, label)
		optimizer.zero_grad()
		loss.backward() 
		optimizer.step()
		print (loss)


		# if i % 5 == 0:
		# 	plot(data, label, prediction)









