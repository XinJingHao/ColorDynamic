from utils.PoE import PositionalEncoding_NTD
from utils.TWQ import TimeWindowQueue_NTD
import torch.nn.functional as F
import torch.nn as nn
import torch

def orthogonal_init(layer, gain=1.414):
	for name, param in layer.named_parameters():
		if 'bias' in name:
			nn.init.constant_(param, 0)
		elif 'weight' in name:
			nn.init.orthogonal_(param, gain=gain)
	return layer


class Transqer_networks(nn.Module):
	def __init__(self, opt):
		super(Transqer_networks, self).__init__()
		self.d = opt.state_dim - 8 # s[0:7] is robot state, s[8:] is lidar results
		# Define the Transformer Encoder block(note that state_dim should be a even number):
		self.pe = PositionalEncoding_NTD(maxlen=opt.T, emb_size=self.d) # for (N,T,d)
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=opt.H, dropout=0,
												   dim_feedforward=opt.net_width, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.L)

		self.fc1 = orthogonal_init(nn.Linear(int(self.d + opt.state_dim), opt.net_width))
		self.fc2 = orthogonal_init(nn.Linear(opt.net_width, int(opt.net_width/2)))
		self.fc3 = orthogonal_init(nn.Linear(int(opt.net_width/2), opt.action_dim))

	def forward(self, TW_s):
		'''TW_s.shape = (B,T,D)'''
		temporal_ld = TW_s[:,:,8:]  # s[0:7] is robot state, s[8:] is lidar results
		temporal_ld = self.pe(temporal_ld) # (N,T,d)
		temporal_ld = self.transformer_encoder(temporal_ld) # (N,T,d)
		temporal_ld = temporal_ld.mean(dim=1) # (N,T,d) ->  (N,d)

		aug_s = torch.cat((temporal_ld,TW_s[:,0,:]),dim=-1) # (N,d+S_dim)

		q = F.relu(self.fc1(aug_s)) # (N,256)
		q = F.relu(self.fc2(q)) # (N,128)
		q = self.fc3(q)  # (N,a_dim)
		return q


class Transqer_agent(object):
	'''Only or Evaluation and Play, not for Training'''
	def __init__(self,opt):
		self.action_dim = opt.action_dim
		self.dvc = opt.dvc
		self.N = opt.N

		# Build Transqer
		self.q_net = Transqer_networks(opt).to(self.dvc)

		# vectorized e-greedy exploration
		self.p = torch.ones(opt.N, device=self.dvc) * 0.01

		# temporal window queue for interaction:
		self.queue = TimeWindowQueue_NTD(opt.N, opt.T, opt.state_dim, opt.dvc, padding=0)

	def select_action(self, TW_s, deterministic):
		'''Input: batched state in (N, T, s_dim) on device
		   Output: batched action, (N,), torch.tensor, on device '''
		with torch.no_grad():
			a = self.q_net(TW_s).argmax(dim=-1)
			if deterministic:
				return a
			else:
				replace = torch.rand(self.N, device=self.dvc) < self.p  # [n]
				rd_a = torch.randint(0, self.action_dim, (self.N,), device=self.dvc)
				a[replace] = rd_a[replace]
				return a

	def load(self,steps):
		self.q_net.load_state_dict(torch.load("./model/{}k.pth".format(steps), map_location=self.dvc, weights_only=True))