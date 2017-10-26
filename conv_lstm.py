# reference : https://github.com/rogertrullo/pytorch_convlstm/

import torch.nn as nn
from torch.autograd import Variable
import torch




def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Custom_CLSTM_cell(nn.Module):
	"""Initialize a basic Conv LSTM cell.
	Args:
	  shape: int tuple thats the height and width of the hidden states h and c()
	  filter_size: int that is the height and width of the filters
	  num_features: int thats the num of channels of the states, like hidden_size

	"""
	def __init__(self):
		super(Custom_CLSTM_cell, self).__init__()

		self.encoding = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=3, stride=(1,1,2,2)),
			nn.ELU(inplace=True),
			nn.Conv2d(8, 8, kernel_size=3, stride=(1,1,1,1)),
			nn.ELU(inplace=True),
			nn.Conv2d(8, 8, kernel_size=3, stride=(1,1,2,2)),
			nn.ELU(inplace=True),
			nn.Conv2d(8, 4, kernel_size=3, stride=(1,1,1,1)),
			nn.ELU(inplace=True),
		)

		self.CLSTM = nn.Conv2d(8, 4* 4, kernel_size=3, stride=(1,1,1,1), 1)

		self.decoding = nn.Sequential(
			nn.ConvTranspose2d(4, 8, kernel_size=1, stride=(1,1,1,1)),
			nn.ELU(inplace=True),
			nn.ConvTranspose2d(8, 8, kernel_size=3, stride=(1,1,2,2)),
			nn.ELU(inplace=True),
			nn.ConvTranspose2d(8, 8, kernel_size=3, stride=(1,1,1,1)),
			nn.ELU(inplace=True),
			nn.ConvTranspose2d(8, 3, kernel_size=3, stride=(1,1,2,2)),
		)



	def forward(self, input, hidden_state):
		hidden,c = hidden_state #hidden and c are images with several channels

		encoding = self.encoding(input)

		combined = torch.cat((encoding, hidden), 1) #concatenate in the channels
		B = self.CLSTM(combined)

		(ai,af,ao,ag)=torch.split(B, 4 ,dim=1)#it should return 4 tensors
		i=torch.sigmoid(ai)
		f=torch.sigmoid(af + 1.0) #forget bias = 1
		o=torch.sigmoid(ao)
		g=torch.tanh(ag)

		next_c= f*c+i*g
		next_h=o*torch.tanh(next_c)

		out = self.decoding()

		return next_h, next_c

	def init_hidden(self,batch_size):
		return ( Variable(torch.zeros(batch_size, 4, 8, 8)).cuda(),
			Variable(torch.zeros(batch_size,4,8,8)).cuda())


class CLSTM(nn.Module):
	"""Initialize a basic Conv LSTM cell.
	Args:
	  shape: int tuple thats the height and width of the hidden states h and c()
	  filter_size: int that is the height and width of the filters
	  num_features: int thats the num of channels of the states, like hidden_size

	"""
	def __init__(self, shape, input_chans, filter_size, num_features,num_layers):
		super(CLSTM, self).__init__()

		self.shape = shape#H,W
		self.input_chans=input_chans
		self.filter_size=filter_size
		self.num_features = num_features
		self.num_layers=num_layers
		cell_list=[]
		# cell_list.append(Custm_CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cuda())#the first
		#one has a different number of input channels

		for idcell in xrange(self.num_layers):
			cell_list.append(Custom_CLSTM_cell().cuda())
		self.cell_list=nn.ModuleList(cell_list)


	def forward(self, input, hidden_state):
		"""
		args:
			hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
			input is the tensor of shape seq_len,Batch,Chans,H,W

		"""

		current_input = input.transpose(0, 1)#now is seq_len,B,C,H,W
		#current_input=input
		next_hidden=[]#hidden states(h and c)
		seq_len=current_input.size(0)


		for idlayer in xrange(self.num_layers):#loop for every layer

			hidden_c=hidden_state[idlayer]#hidden and c are images with several channels
			all_output = []
			output_inner = []
			for t in xrange(seq_len):#loop for every step
				hidden_c=self.cell_list[idlayer](current_input[t,...],hidden_c)#cell_list is a list with different conv_lstms 1 for every layer

				output_inner.append(hidden_c[0])

			next_hidden.append(hidden_c)
			current_input = torch.cat(output_inner, 0).view(current_input.size(0), *output_inner[0].size())#seq_len,B,chans,H,W


		return next_hidden, current_input

	def init_hidden(self,batch_size):
		init_states=[]#this is a list of tuples
		for i in xrange(self.num_layers):
			init_states.append(self.cell_list[i].init_hidden(batch_size))
		return init_states


