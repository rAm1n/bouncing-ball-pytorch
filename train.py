
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import bouncing_balls as b
from conv_lstm import CLSTM, weights_init
import numpy as np
import cv2
import time



num_features=10
filter_size=3
batch_size=4
shape=(32,32) #H,W
inp_chans=3
nlayers=1
seq_len=10
num_balls = 2
max_step = 200000
seq_start = 5
lr = .001
keep_prob = 0.8
dtype = torch.cuda.FloatTensor
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
	dat = np.zeros((batch_size, seq_length, shape, shape, 3), dtype=np.float32)
	for i in xrange(batch_size):
		dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
	return torch.from_numpy(dat).permute(0,1,4,2,3)



def train():
	model = CLSTM(shape, inp_chans, filter_size, num_features,nlayers)
	model.apply(weights_init)
	model = model.cuda()

	crit = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	drop = nn.Dropout(keep_prob)
	hidden_state = model.init_hidden(batch_size)


	start = time.time()
	l = list()
	for step in xrange(max_step):
		dat = generate_bouncing_ball_sample(batch_size, seq_len, shape[0], num_balls)
		input = Variable(dat.cuda(), requires_grad=True)
		input = drop(input)
		target = Variable(dat.cuda(), requires_grad=False)
		hidden_state = model.init_hidden(batch_size)

		output = list()
		for i in xrange(input.size(1)-1):
			if i < seq_start:
				out , hidden_c = model(input[:,i,:,:,:].unsqueeze(1), hidden_state)
			else:
				out , hidden_c = model(out, hidden_state)
			output.append(out)

		output = torch.cat(output,1)
		loss = crit(output[:,seq_start:,:,:,:], target[:,seq_start+1:,:,:,:])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		l.append(loss.data[0])

		if step%100 == 0 and step != 0:
			print(np.array(l).mean(), time.time()-start)
			l = list()
			start = time.time()

		if step%1000 == 0:
			# make video
			print(step)
			print("now generating video!")
			video = cv2.VideoWriter()
			success = video.open("generated_conv_lstm_video_{0}.avi".format(step), fourcc, 4, (180, 180), True)
			hidden_state = model.init_hidden(batch_size)
			output = list()
			for i in xrange(25):
				if i < seq_start:
					out , hidden_c = model(input[:,i,:,:,:].unsqueeze(1), hidden_state)
				else:
					out , hidden_c = model(out, hidden_state)
				output.append(out)
			ims = torch.cat(output,1).permute(0,1,4,3,2)
			ims = ims[0].data.cpu().numpy()
			for i in xrange(5,25):
				x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
				new_im = cv2.resize(x_1_r, (180,180))
				video.write(new_im)
			video.release()




train()

