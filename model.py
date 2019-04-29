from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as ptv
# class (pt.nn.Module):
# 	def __init__(self, layers input_dim, output_dim):
# 		super(MLP, self).__init__()
# 		self.layers = layers
# 		self.input_dim = input_dim
# 		self.output_dim = output_dim
class Net(torch.nn.Module): 
	def __init__(self, input_dim, hidden1, hidden2, hidden_pred, output_dim):
		super(Net, self).__init__()   
		self.hidden1 = torch.nn.Linear(input_dim, hidden1) 
		self.hidden2 = torch.nn.Linear(hidden1, hidden2)
		self.hidden3 = torch.nn.Linear(hidden2, hidden_pred)
		# self.predict = torch.nn.Linear(hidden_pred, output_dim) # 输出层线性输出 
	def __call__(self, din):
		return self.forward(din)
	def forward(self, din):
		din = din.view(-1, din.shape[1])
		x = F.sigmoid(self.hidden1(din))
		x = F.sigmoid(self.hidden2(x))
		x = F.sigmoid(self.hidden3(x))
		# prediction = self.predict(x)
		return x
class Encoder(torch.nn.Module):
	def __init__(self, factor_dim, hidden_dim, time_step, num_layer):
		super(Encoder, self).__init__()
		self.factor_dim = factor_dim
		self.hidden_dim = hidden_dim
		self.tiem_step = time_step
		self.num_layer = num_layer
		self.lstm_layer = torch.nn.LSTM(self.factor_dim, self.hidden_dim, self.num_layer, bidirectional = False)
		self.prediction = torch.nn.Linear(hidden_dim, 1)
	def __call__(self, input_data):
		return self.forward(input_data)
	def forward(self, input_data):
		batch = input_data.shape[1]
		h0 = torch.randn(self.num_layer, batch, self.hidden_dim)
		c0 = torch.randn(self.num_layer, batch, self.hidden_dim)
		out, (hn, cn) = self.lstm_layer(input_data, (h0, c0))
		prediction = out[-1]
		prediction = self.prediction(prediction)
		return prediction
# class Attention(torch.nn.Module):
# 	def __init__(self, hidden_dim, encoder_dim):
# 		super(Attention, self).__init__()
# 		self.hidden_dim = hidden_dim
# 		self.encoder_dim = encoder_dim
# 		self.Linearforward1 = torch.nn.Linear(self.encoder_dim, self.hidden_dim)
# 	def __call__(self, encoder_output, hn, cn):
# 		return self.forward(encoder_output, hn, cn)
# 	def forward(self, encoder_output, hn, cn):
# 		weights_h = self.Linearforward1(encoder_output)
# 		weights_h = torch.nn.functional.tanh(weights_h)
# 		weights_h = torch.mv(weights_h, hn[-1])
# 		weights_h = torch.nn.functional.softmax(weights_h, dim=0)
# 		context = torch.mv(encoder_output.t(), weights_h)
# 		#hidden_dim
# 		return context



# class Decoder(torch.nn.Module):
# 	def __init__(self, input_dim, encoder_dim, hidden_dim, time_step, num_layer):
# 		super(Decoder, self).__init__()
# 		self.hidden_dim = hidden_dim
# 		self.time_step = time_step
# 		self.num_layer = num_layer
# 		self.lstm_layer = torch.nn.LSTM(1, self.hidden_dim, num_layer)
# 		self.fc = torch.nn.Linear(encoder_dim + 1, 1)
# 		self.fc_final = nn.Linear(hidden_dim + encoder_dim, 1)
# 		self.fc.weight.data.normal_()
# 	def __call__(self, input, hn, cn, label):
# 		return self.forward(input, hn, cn, label)
# 	def forward(self, input, hn, cn, label):


	





