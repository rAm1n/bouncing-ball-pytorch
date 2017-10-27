
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
dtype = torch.cuda.FloatTensor
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
	dat = np.zeros((batch_size, seq_length, shape, shape, 3), dtype=np.float32)
	for i in xrange(batch_size):
		dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
	return torch.from_numpy(dat).permute(0,1,4,2,3)




#If using this format, then we need to transpose in CLSTM
# input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()


# dat = Variable(batch_size, seq_len, inp_chans, shape[0], shape[1])


# dat = generate_bouncing_ball_sample(batch_size,seq_len, 32, 2)
# dat = torch.from_numpy(dat).transpose(4,3).transpose(3,2).float()

# input = Variable(dat).cuda()

# conv_lstm=CLSTM(shape, inp_chans, filter_size, num_features,nlayers)
# conv_lstm.apply(weights_init)
# conv_lstm.cuda()

# print 'convlstm module:',conv_lstm


# print 'params:'
# params=conv_lstm.parameters()
# #for p in params:
# #   print 'param ',p.size()
# #   print 'mean ',torch.mean(p)


# hidden_state=conv_lstm.init_hidden(batch_size)
# print 'hidden_h shape ',len(hidden_state)
# print 'hidden_h shape ',hidden_state[0][0].size()
# out=conv_lstm(input,hidden_state)
# print 'out shape',out[0].size()
# print 'len hidden ', len(out[1])
# print 'next hidden',out[1][0][0].size()
# print 'convlstm dict',conv_lstm.state_dict().keys()


# L=torch.sum(out[0])
# L.backward()


def train():
	global model, input,dat , out,output
	model = CLSTM(shape, inp_chans, filter_size, num_features,nlayers)
	model.apply(weights_init)
	model = model.cuda()

        crit = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)		

#	input = Variable(torch.zeros(batch_size,seq_len,inp_chans,shape[0],shape[1]).type(dtype).cuda())
	hidden_state = model.init_hidden(batch_size)


	start = time.time()
	l = list()
	for step in xrange(max_step):
		dat = generate_bouncing_ball_sample(batch_size, seq_len, shape[0], num_balls)
		input = Variable(dat.cuda(), requires_grad=True)
		target = Variable(dat.cuda(), requires_grad=False)
 	
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
		


		# if step%100 == 0 and step != 0:
		# 	# summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
		# 	# summary_writer.add_summary(summary_str, step)
		# 	# print("time per batch is " + str(elapsed))
		# 	# print(step)
		# 	# print(loss_r)
		# 	print(time.time()-start)
		# 	start = time.time()

		# assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

		# if step%1000 == 0:
		# 	# checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
		# 	# saver.save(sess, checkpoint_path, global_step=step)
		# 	# print("saved to " + FLAGS.train_dir)

		# 	# make video
		# 	print("now generating video!")
		# 	video = cv2.VideoWriter()
		# 	success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
		# 	dat_gif = dat
		# 	ims = sess.run([x_unwrap_g],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
		# 	ims = ims[0][0]
		# 	print(ims.shape)
		# 	for i in xrange(50 - FLAGS.seq_start):
		# 		x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
		# 		new_im = cv2.resize(x_1_r, (180,180))
		# 		video.write(new_im)
		# 	video.release()




train()

