
import torchvision
import torch
from torch import autograd
from torch.nn.Module import Variable
import torch.nn.functional as F
import bouncing_balls as b
from Custom_CLSTM_cell import CLSTM, weights_init



def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
	dat = np.zeros((batch_size, seq_length, shape, shape, 3))
	for i in xrange(batch_size):
		dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
	return dat




num_features=10
filter_size=3
batch_size=4
shape=(32,32)#H,W
inp_chans=3
nlayers=2
seq_len=10

#If using this format, then we need to transpose in CLSTM
# input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()

input = Variable(generate_bouncing_ball_sample(batch_size,seq_len, 32, 2)).cuda()

conv_lstm=CLSTM(shape, inp_chans, filter_size, num_features,nlayers)
conv_lstm.apply(weights_init)
conv_lstm.cuda()

print 'convlstm module:',conv_lstm


print 'params:'
params=conv_lstm.parameters()
for p in params:
   print 'param ',p.size()
   print 'mean ',torch.mean(p)


hidden_state=conv_lstm.init_hidden(batch_size)
print 'hidden_h shape ',len(hidden_state)
print 'hidden_h shape ',hidden_state[0][0].size()
out=conv_lstm(input,hidden_state)
print 'out shape',out[1].size()
print 'len hidden ', len(out[0])
print 'next hidden',out[0][0][0].size()
print 'convlstm dict',conv_lstm.state_dict().keys()


L=torch.sum(out[1])
L.backward()
