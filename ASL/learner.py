import os
import time
import torch
from copy import deepcopy
from utils.Transqer import Transqer_networks


def learner_process(opt):
	learner = Learner(opt)
	learner.run()

class Learner:
	def __init__(self, opt):
		self.L_dvc = torch.device(opt.L_dvc)
		self.shared_data = opt.shared_data
		self.max_train_steps = opt.max_train_steps
		self.explore_steps = opt.explore_steps
		self.gamma = opt.gamma
		self.clip = opt.clip
		self.lr = opt.lr

		self.upload_freq = opt.upload_freq
		self.Bstep = 0 # Learner进行Backpropagation的次数
		self.save_freq = opt.save_freq # in Bstep
		self.batch_size = opt.batch_size
		self.TPS = opt.TPS
		self.N = opt.N

		'''Time Feedback Mechanism Configuration'''
		self.time_feedback = opt.time_feedback
		self.tf_rho = opt.N * opt.TPS / opt.batch_size # 使用tf时，进行一次Vstep的同时，应进行rho次Bstep --- Eq.(2) of the Color paper

		'''Build Double Q-Transformer'''
		self.q_net = Transqer_networks(opt).to(self.L_dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
		self.upload_model()

		self.q_target = Transqer_networks(opt).to(self.L_dvc)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.tau = 0.005
		self.soft_target = opt.soft_target

		'''symmetric invariance'''
		self.action_mapping = torch.tensor([4,3,2,1,0,5,6,7], device=self.L_dvc)
		if opt.compile: self.expanding_with_SI = torch.compile(self.expanding_with_SI)

		if not os.path.exists('model'): os.mkdir('model')
		print('Learner Started!')

	def run(self):
		mean_t = 0 # average time of bp once
		while True:
			total_steps = self.shared_data.get_total_steps()
			if total_steps > self.max_train_steps:
				print('---------------- Learner Finished ----------------')
				break #结束Learner进程

			if total_steps < self.explore_steps:
				time.sleep(0.2)
			else:
				t0 = time.time()
				'''训练模型'''
				self.train()
				self.Bstep += 1

				'''保存模型至硬盘'''
				if self.Bstep % self.save_freq == 0:
					self.save_model(total_steps)

				'''更新target网络'''
				if self.soft_target:
					self.soft_target_update() # maybe more stable
				elif self.Bstep % int(1/self.tau) == 0:
					self.hard_target_update() # much faster than soft target

				'''上传模型至sharer'''
				if self.Bstep % self.upload_freq == 0:
					self.upload_model()
					self.shared_data.set_should_download(True) #inform actor to download latest model
					# print('(Learner) Upload model to the sharer. ')

				'''时间反馈机制'''
				if self.time_feedback:
					# 计算
					current_t = time.time() - t0 #本次训练消耗的时间
					mean_t = mean_t + (current_t-mean_t)/self.Bstep #增量法求得的Bstep平均时间
					scalled_learner_time = self.tf_rho * mean_t
					# 使用tf时，进行一次Vstep的同时，应进行rho次Bstep ---- Eq.(2) of the Color paper
					# 因此，一次Vstep的时间应该约等于rho次Bstep的时间 ---- Eq.(4) of the Color paper
					# 当 t[Vstep] < rho * t[Bstep]时，表明actor太快。这种情况下，每次Vstep时，actor等待 (rho * t[Bstep] - t[Vstep]) 秒
					# 当 t[Vstep] > rho * t[Bstep]时，表明learner太快。这种情况下，每次Bstep时，learner等待 (t[Vstep] - t[Bstep])/rho 秒

					self.shared_data.set_t(scalled_learner_time, 1) # 存储,将learner时间放在第1位
					# 比较、等待
					t = self.shared_data.get_t()
					if t[1]<t[0]:
						hold_time = (t[0]-t[1])/self.tf_rho
						if hold_time > 1: hold_time = 1
						time.sleep( hold_time ) #learner耗时少，则learner等待(分成tf_rho次等待)

					if self.Bstep % 5000 == 0:
						print(f'(Learner) Total steps:{int(total_steps/1000)}k ; Target TPS:{self.TPS}; Real TPS:{round(self.batch_size * self.Bstep / (total_steps-self.explore_steps),2)}')

						# 耗时情况
						Actor_Time = round(t[0],3)
						Learner_Time = round(t[1],3)
						Await_Time = round(abs(t[0]-t[1]),4)
						Consumed_Time_per_Transition = round(max(Actor_Time, Learner_Time)/self.N,4)
						print(f'(Learner) Actor Time:{Actor_Time}s ; Learner Time:{Learner_Time}s ; Await Time:{Await_Time}s; '
							  f'Consumed Time per Transition:{Consumed_Time_per_Transition}s') # 该值越低，整个框架所需的训练时间越短
						print(f'(Learner) Predicted total running time:{round(Consumed_Time_per_Transition*self.max_train_steps/3600,1)}h\n')

	def state_projection(self, state):
		"""Expand the state using Symmetric Invariance(SI), input shape [B,T,S_dim]"""
		s = state.clone()
		s[:,:,[1,3,5,7]] *= -1 # orientation related
		s[:,:,8:] = s[:,:,8:].flip(dims=[2]) # LiDAR related
		return s

	def action_projection(self, action):
		"""Expand the action using Symmetric Invariance(SI), input shape [B,1]
		SI: [0,1,2,3,4,5,6,7] -> [4,3,2,1,0,5,6,7]"""
		return self.action_mapping[action]

	def expanding_with_SI(self, s, a, r, s_next, dw, ct):
		'''Expanding training transitions using Symmetric Invariance(SI)'''
		s = torch.cat((s, self.state_projection(s)), dim=0) # (2*B,D,T)
		a = torch.cat((a, self.action_projection(a)), dim=0)
		r = torch.cat((r, r), dim=0)
		s_next = torch.cat((s_next, self.state_projection(s_next)), dim=0)  # (2*B,D,T)
		dw = torch.cat((dw, dw), dim=0)
		ct = torch.cat((ct, ct), dim=0)
		return s, a, r, s_next, dw, ct

	def train(self):
		s, a, r, s_next, dw, ct = self.shared_data.sample() # on self.L_dvc
		s, a, r, s_next, dw, ct = self.expanding_with_SI(s, a, r, s_next, dw, ct) # Symmetric Invariance

		'''Compute the target Q value with Double Q-learning'''
		with torch.no_grad():
			argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
			max_q_next = self.q_target(s_next).gather(1,argmax_a)
			target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

		'''Get current Q estimates'''
		current_q_a = self.q_net(s).gather(1,a)

		'''Mse regression'''
		q_loss = torch.square(ct * (current_q_a - target_Q)).mean() # drop the inconsistant transitions
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip)
		self.q_net_optimizer.step()

		'''Update the target net'''
		if self.soft_target:
			# soft target update
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			# hard target update
			if self.Bstep % int(1/self.tau) == 0:
				for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
					target_param.data.copy_(param.data)

	def hard_target_update(self):
		'''HardTargetASL+SparrowV1+16trainmap无法学习？'''
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(param.data)
			target_param.requires_grad = False

	def soft_target_update(self):
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			target_param.requires_grad = False


	def upload_model(self):
		'''好像不是很高效，如何优化？'''
		self.shared_data.set_net_param(deepcopy(self.q_net).cpu().state_dict()) # 这里必须deepcopy，否则会将learner模型放到cpu上去

	def save_model(self, total_steps):
		torch.save(self.q_net.state_dict(), f"./model/{int(total_steps / 1e3)}k.pth")


