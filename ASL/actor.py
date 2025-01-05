from Sparrow_V2 import Sparrow
from utils.Transqer import Transqer_networks
from utils.Scheduler import LinearSchedule
from utils.TWQ import TimeWindowQueue_NTD
import torch
import time


def actor_process(opt):
	actor = Actor(opt)
	actor.run()

class Actor:
	def __init__(self, opt):
		#basic information init
		self.A_dvc = torch.device(opt.A_dvc)

		self.O = opt.O
		self.N = opt.N
		self.reset_freq = opt.reset_freq
		self.exp_name = opt.exp_name
		self.shared_data = opt.shared_data
		self.state_dim = opt.state_dim
		self.action_dim = opt.action_dim
		self.explore_steps = opt.explore_steps
		self.time_feedback = opt.time_feedback
		self.max_train_steps = opt.max_train_steps

		# vectorized e-greedy exploration
		self.explore_frac_scheduler = LinearSchedule(opt.decay_step, opt.init_explore_frac, opt.end_explore_frac)
		self.p = torch.ones(opt.N, device=self.A_dvc)
		self.min_eps = opt.min_eps

		# build vectorized envs and actor
		self.envs = Sparrow(**vars(opt))
		self.actor_net = Transqer_networks(opt).to(self.A_dvc)
		for p in self.actor_net.parameters(): p.requires_grad = False

		# temporal window queue:
		self.queue = TimeWindowQueue_NTD(opt.N, opt.T, opt.state_dim, device=self.A_dvc, padding=0)

		# total interacting steps (+=N)
		self.total_steps = 0

		self.t_start = time.time()
		print('Actor Started!')


	def run(self):
		s, info = self.envs.reset()
		ct = torch.ones(self.N,device=self.A_dvc,dtype=torch.bool)
		self.queue.clear()

		ep_r, mean_t, c = 0, 0, 0
		while True:
			if self.total_steps > self.max_train_steps:
				print('---------------- Actor Finished ----------------')
				break #结束Actor进程

			'''向sharer更新total steps'''
			self.total_steps += self.N
			self.shared_data.set_total_steps(self.total_steps)

			'''记录本次交互起始时间(For TFM)'''
			if self.total_steps > self.explore_steps: t0 = time.time()

			'''Baby step课程学习'''
			if self.total_steps % (self.reset_freq * self.N) == 0: # Vsteps -> Tsteps
				self.envs.O = int(self.O * min(0.15,self.total_steps/self.max_train_steps)/0.15)+1  # 前15%进行baby step学习
				print(f'(Actor) {self.exp_name}, Total steps: {round(self.total_steps / 1e3, 2)}k; Obstacle Numbers: {self.envs.O}')

				s, info = self.envs.reset()
				ct.fill_(True)
				self.queue.clear()
				continue


			"""单步state -> 时序窗口state"""
			self.queue.append(s) # 将s加入时序窗口队列
			TW_s = self.queue.get() # 取出队列所有数据 (N,T,D)

			"""交互"""
			a = self.select_action(TW_s, deterministic=False)
			s_next, r, dw, tr, info = self.envs.step(a)

			"""保存数据，更新length, ct, s"""
			self.shared_data.add(TW_s, a, r, dw, ct) #注意ct是用上一次step的， 表示s与s_next是否来自同一条轨迹； (s1,a1,r2,dw2,ct1)
			self.queue.padding_with_done(~ct)  # 根据上一时刻的ct去padding
			ct = ~(dw + tr)  # 如果当前s_next是”截断状态“或”终止状态“，则s_next与s_next_next是不consistent的，训练时要丢掉
			s = s_next

			'''打印回合累计奖励'''
			ep_r += r[0]
			if dw[0] or tr[0]:
				print('-------------------------------------------------------------------------------------------------')
				print(f'(Actor) {self.exp_name}, N{self.N}, Total steps: {round(self.total_steps / 1e3, 2)}k, ep_r: {round(ep_r.item(),1)}')
				ep_r = 0
				time_consumed = time.time() - self.t_start
				print(f'(Actor) Consumed Time: {round(time_consumed/3600,1)}h ({round(time_consumed/60,1)}min)')
				print('-------------------------------------------------------------------------------------------------\n')

			if self.total_steps > self.explore_steps:
				'''download model parameters from shared_data.net_param'''
				if self.total_steps % (10*self.N) == 0: # don't ask shared_data too frequently
					if self.shared_data.get_should_download():
						self.shared_data.set_should_download(False)
						self.download_model()
						# print('(Actor) Download model from sharer.')

				'''fresh vectorized e-greedy noise every 3.2k total steps'''
				if self.total_steps % (int(3200/self.N)*self.N) == 0:
					self.fresh_explore_prob(self.total_steps)

				'''Time Feedback Mechanism'''
				# 使用tf时，进行一次Vstep的同时，应进行rho次Bstep ---- Eq.(2) of the Color paper
				# 因此，一次Vstep的时间应该约等于rho次Bstep的时间 ---- Eq.(4) of the Color paper
				# 当 t[Vstep] < rho * t[Bstep]时，表明actor太快。这种情况下，每次Vstep时，actor等待 (rho * t[Bstep] - t[Vstep]) 秒
				# 当 t[Vstep] > rho * t[Bstep]时，表明learner太快。这种情况下，每次Bstep时，learner等待 (t[Vstep] - t[Bstep])/rho 秒
				if self.time_feedback:
					# 计算
					c += 1
					current_t = time.time() - t0  # 本次step消耗的时间
					mean_t = mean_t + (current_t - mean_t) / c  # 增量法求得的平均step时间
					# 存储
					self.shared_data.set_t(mean_t, 0) # actor时间放在第0位
					# 比较、等待
					t = self.shared_data.get_t()
					if t[0]<t[1]:
						hold_time = t[1]-t[0]
						if hold_time > 1: hold_time = 1
						time.sleep(hold_time) #actor耗时少，则actor等待


	def fresh_explore_prob(self, steps):
		#fresh vectorized e-greedy noise
		explore_frac = self.explore_frac_scheduler.value(steps)
		i = int(explore_frac * self.N)
		explore = torch.arange(i, device=self.A_dvc) / (2 * i)  # 0 ~ 0.5

		self.p.fill_(self.min_eps)
		self.p[self.N - i:] += explore
		self.p = self.p[torch.randperm(self.N)]  # 打乱vectorized e-greedy noise, 让探索覆盖每一个地图

	def select_action(self, TW_s, deterministic):
		'''Input: batched state in (N, T, s_dim) on device
		   Output: batched action, (N,), torch.tensor, on device '''
		with torch.no_grad():
			a = self.actor_net(TW_s).argmax(dim=-1)
			if deterministic:
				return a
			else:
				replace = torch.rand(self.N, device=self.A_dvc) < self.p  # [n]
				rd_a = torch.randint(0, self.action_dim, (self.N,), device=self.A_dvc)
				a[replace] = rd_a[replace]
				return a

	def download_model(self):
		self.actor_net.load_state_dict(self.shared_data.get_net_param())
		for p in self.actor_net.parameters(): p.requires_grad = False
