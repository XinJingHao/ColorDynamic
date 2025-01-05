import torch
import time


class shared_data():
	def __init__(self, opt):
		self.A_dvc = torch.device(opt.A_dvc)
		self.B_dvc = torch.device(opt.B_dvc)
		self.L_dvc = torch.device(opt.L_dvc)

		self.max_size = int(opt.buffersize/opt.N)
		self.state_dim = opt.state_dim
		self.N = opt.N
		self.ptr = 0
		self.size = 0
		self.full = False
		self.batch_size = opt.batch_size

		# init shared buffer
		self.s = torch.zeros((self.max_size, opt.N, opt.T, opt.state_dim), device=self.B_dvc) # TW_s
		self.a = torch.zeros((self.max_size, opt.N, 1), dtype=torch.int64, device=self.B_dvc)
		self.r = torch.zeros((self.max_size, opt.N, 1), device=self.B_dvc)
		self.dw = torch.zeros((self.max_size, opt.N, 1), dtype=torch.bool, device=self.B_dvc)
		self.ct = torch.zeros((self.max_size, opt.N, 1),dtype=torch.bool, device=self.B_dvc)

		# init shared data
		self.t = [0,0] # time feedback, 0是actor时间，1是scalled的learner时间
		self.net_param = None # net.state_dict(), upload/download model
		self.total_steps = 0
		self.should_download = False  # whether actor should download model now

		#Tread lock
		self.get_lock_time = 2e-4
		self.set_lock_time = 1e-4
		self.busy = [False, False] #标记某个共享数据是否正在被占用, F表示空闲，T表示正在get()/set()
		# self.busy[0] 标记net_param
		# self.busy[1] 标记buffer

		print('Sharer Started!')

	def add(self, s, a, r, dw, ct):
		'''add transitions to buffer,with thread lock'''
		self.set_lock(self.add_core, 1, (s, a, r, dw, ct))  # use self.busy[1] to lock buffer data

	def add_core(self, trans):
		'''add transitions to buffer,without thread lock'''
		s, a, r, dw, ct = trans # on self.A_dvc

		if self.A_dvc != self.B_dvc:
			s = s.to(self.B_dvc)
			a = a.to(self.B_dvc)
			r = r.to(self.B_dvc)
			dw = dw.to(self.B_dvc)
			ct = ct.to(self.B_dvc)

		self.s[self.ptr] = s # TW_s, shape=(N, T, state_dim); Batched TimeWindow state
		self.a[self.ptr] = a.unsqueeze(-1)  #(N,) to (N,1)
		self.r[self.ptr] = r.unsqueeze(-1)
		self.dw[self.ptr] = dw.unsqueeze(-1)
		self.ct[self.ptr] = ct.unsqueeze(-1)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		if self.size == self.max_size:
			self.full = True

	def sample(self):
		'''sample batch transitions, with threading lock'''
		return self.get_lock(self.sample_core, 1)  # use self.busy[1] to lock buffer data

	def sample_core(self):
		'''sample batch transitions, without thread lock'''
		if not self.full:
			ind = torch.randint(low=0, high=self.ptr - 1, size=(self.batch_size,), device=self.B_dvc)  # sample from [0, ptr-2]
		else:
			ind = torch.randint(low=0, high=self.size - 1, size=(self.batch_size,), device=self.B_dvc)  # sample from [0, size-2]

			# # 忽略下面的问题，保证sample出的batch_size始终不变，以保证learner里的SI可以compile
			# if self.ptr - 1 in ind: ind = ind[ind != (self.ptr - 1)] # delate ptr - 1 in [0, size-2]

		env_ind = torch.randint(low=0, high=self.N, size=(len(ind),), device=self.B_dvc) # [l,h)
		# [N, T, s_dim], [N, 1], [N, 1], [N, T, s_dim], [N, 1], [N, 1]

		if self.B_dvc != self.L_dvc:
			return (self.s[ind,env_ind].to(self.L_dvc), self.a[ind,env_ind].to(self.L_dvc),
					self.r[ind,env_ind].to(self.L_dvc), self.s[ind + 1,env_ind].to(self.L_dvc),
					self.dw[ind,env_ind].to(self.L_dvc), self.ct[ind, env_ind].to(self.L_dvc))
		else:
			return (self.s[ind, env_ind], self.a[ind, env_ind], self.r[ind, env_ind],
					self.s[ind + 1, env_ind], self.dw[ind, env_ind], self.ct[ind, env_ind])


	def get_net_param(self):
		return self.get_lock(self.get_net_param_core, 0) # use self.busy[0] to lock net_param
	def get_net_param_core(self):
		return self.net_param

	def set_net_param(self, net_param):
		self.set_lock(self.set_net_param_core, 0, net_param) # use self.busy[0] to lock net_param
	def set_net_param_core(self, net_param):
		self.net_param = net_param

	#---------------------------------下面是没加进程锁的函数----------------------------#
	# Time feedback
	def get_t(self):
		return self.t
	def set_t(self,time,idx):
		self.t[idx] = time

	# 总交互次数
	def get_total_steps(self):
		return self.total_steps
	def set_total_steps(self,total_steps):
		self.total_steps = total_steps

	# Actor是否应该下载模型
	def get_should_download(self):
		return self.should_download
	def set_should_download(self, bol):
		self.should_download = bol

	# 进程锁
	def get_lock(self, get_func, idx):
		''' get_func is the function to be lock, idx is the index of self.busy '''
		while True:
			if self.busy[idx]:
				time.sleep(self.get_lock_time) #等待
			else:
				time.sleep(self.get_lock_time)  #Double check,防止同时占用 (get/set have different double check freq)
				if not self.busy[idx]:
					self.busy[idx] = True # 占用
					data = get_func() #被锁的操作
					self.busy[idx] = False  # 解除占用
					return data

	def set_lock(self, set_func, idx, data):
		''' set_func is the function to be lock, idx is the index of self.busy, data is the data to be set '''
		while True:
			if self.busy[idx]:
				time.sleep(self.set_lock_time)  # 等待
			else:
				time.sleep(self.set_lock_time)  # 以get()不同的频率再次check,防止同时占用
				if not self.busy[idx]:
					self.busy[idx] = True # 占用
					set_func(data) #被锁的操作
					self.busy[idx] = False # 解除占用
					break
