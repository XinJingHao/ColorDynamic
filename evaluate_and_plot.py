from torch.utils.tensorboard import SummaryWriter
from Sparrow_V2 import Sparrow, str2bool
from utils.Transqer import Transqer_agent
from datetime import datetime
import os,shutil
import argparse
import torch


parser = argparse.ArgumentParser()
'''Hyperparameter Setting for Evaluation: model_eval_turns = C*N'''
parser.add_argument('--C', type=int, default=10, help='number of reset times')
parser.add_argument('--N', type=int, default=10, help='number of vectorized environments')

'''Hyperparameter Setting for Transqer'''
parser.add_argument('--net_width', type=int, default=64, help='Linear net width')
parser.add_argument('--T', type=int, default=10, help='length of time window')
parser.add_argument('--H', type=int, default=8, help='Number of Head')
parser.add_argument('--L', type=int, default=3, help='Number of Transformer Encoder Layers')

'''Hyperparameter Setting for Sparrow'''
parser.add_argument('--dvc', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--action_type', type=str, default='Discrete', help='Action type: Discrete / Continuous')
parser.add_argument('--window_size', type=int, default=800, help='size of the map')
parser.add_argument('--D', type=int, default=400, help='maximal local planning distance:366*1.414')
# parser.add_argument('--N', type=int, default=1, help='number of vectorized environments')
parser.add_argument('--O', type=int, default=15, help='number of obstacles in each environment')
parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
parser.add_argument('--RdOV', type=str2bool, default=True, help='whether to randomize the Velocity of dynamic obstacles')
parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
parser.add_argument('--Obs_R', type=int, default=14, help='maximal obstacle radius, cm')
parser.add_argument('--Obs_V', type=int, default=50, help='maximal obstacle velocity, cm/s')
parser.add_argument('--MapObs', type=str, default=None, help="name of map file, e.g. 'map.png' or None")
parser.add_argument('--ld_a_range', type=int, default=360, help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=300, help='max scanning distance of lidar (cm)')
parser.add_argument('--ld_num', type=int, default=72, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--basic_ctrl_interval', type=float, default=0.1, help='control interval (s), 0.1 means 10 Hz control frequency')
parser.add_argument('--ctrl_delay', type=int, default=0, help='control delay, in basic_ctrl_interval, 0 means no control delay')
parser.add_argument('--K', type=tuple, default=(0.55,0.6), help='K_linear, K_angular')
parser.add_argument('--draw_auxiliary', type=str2bool, default=False, help='whether to draw auxiliary infos')
parser.add_argument('--render_speed', type=str, default='fast', help='fast / slow / real')
parser.add_argument('--max_ep_steps', type=int, default=500, help='maximum episodic steps')
parser.add_argument('--noise', type=str2bool, default=True, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=True, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(3.2e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=True, help='whether to use torch.compile to boost simulation speed')
opt = parser.parse_args()
opt.render_mode = None
opt.dvc = torch.device(opt.dvc)


def main():
	# Build env for evaluation
	eval_envs = Sparrow(**vars(opt))
	opt.state_dim = eval_envs.state_dim
	opt.action_dim = eval_envs.action_dim

	# Init Transqer agent
	agent = Transqer_agent(opt)

	# use SummaryWriter to record the training curve
	timenow = str(datetime.now())[0:-10]
	timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
	writepath = f'runs/ColorDynamic-C{opt.C}-N{opt.N}-' + timenow
	if os.path.exists(writepath): shutil.rmtree(writepath)
	writer = SummaryWriter(log_dir=writepath)

	results = []
	for model_name in sorted(os.listdir('model'),key=lambda x:int(x[0:-5])):
		model_idx = int(model_name[0:-5])
		agent.load(model_idx)

		'''Model evaluation'''
		# ----------------------- 测试C种不同的地图，每次开N个并行环境，每个模型共evaluateC*N次 -----------------------
		ep_steps, ep_r, arrival_rate = 0, 0, 0
		for _ in range(opt.C):
			temp_ep_steps, temp_ep_r, temp_arrival_rate = vectorized_model_evaluation(eval_envs, agent, deterministic=False)
			ep_steps += temp_ep_steps
			ep_r += temp_ep_r
			arrival_rate += temp_arrival_rate
		ep_steps /= opt.C
		ep_r /= opt.C
		arrival_rate /= opt.C
		# ----------------------- 测试C种不同的地图，每次开N个并行环境，每个模型共evaluateC*N次 -----------------------

		'''Record raw infos'''
		writer.add_scalar('ep_steps', ep_steps, global_step=model_idx)
		writer.add_scalar('ep_r', ep_r, global_step=model_idx)
		writer.add_scalar('arrival_rate', arrival_rate, global_step=model_idx)

		'''Record normalized score'''
		normed_ep_steps = round((opt.max_ep_steps - ep_steps)/(opt.max_ep_steps-70), 3) # 回合最大步长500, 最快(大约)70步到达
		normed_ep_r = round(ep_r / 220, 3) # 回合累计奖励最大(大约)220
		arrival_rate = round(arrival_rate, 3)
		normed_total_score = round((normed_ep_steps + normed_ep_r + arrival_rate)/3, 3)
		results.append((model_name, normed_total_score, arrival_rate, normed_ep_steps, normed_ep_r))
		print(f"model: {model_name}")
		print(f"Episodic Steps: {int(ep_steps)}, Episodic Rewards: {int(ep_r)}, Arrival Rate: {arrival_rate} \n")
		print(f"Total Score: {normed_total_score}, Arrival Rate: {arrival_rate}, Step Score: {normed_ep_steps}, Reward Score: {normed_ep_r}")
		print("----------------------------------------------------------------------------------------------")

	write('TotalRank.txt',reversed(sorted(results, key=lambda x: x[1])))
	write('ArrivalRank.txt', reversed(sorted(results, key=lambda x: x[2])))
	write('StepRank.txt', reversed(sorted(results, key=lambda x: x[3])))
	write('RewardRank.txt', reversed(sorted(results, key=lambda x: x[4])))

	eval_envs.close()


def vectorized_model_evaluation(envs, agent, deterministic):
	step_collector, total_steps = torch.zeros(opt.N, device=opt.dvc), 0
	r_collector, total_r  = torch.zeros(opt.N, device=opt.dvc), 0
	arrived_vec = torch.zeros(opt.N, dtype=torch.bool, device=opt.dvc)
	finished_vec = torch.zeros(opt.N, dtype=torch.bool, device=opt.dvc)
	finished = 0

	agent.queue.clear()
	s, info = envs.reset()
	ct = torch.ones(opt.N, device=opt.dvc, dtype=torch.bool)
	while not finished_vec.all():
		'''单步state -> 时序窗口state:'''
		agent.queue.append(s)  # 将s加入时序窗口队列
		TW_s = agent.queue.get()  # 取出队列所有数据及

		a = agent.select_action(TW_s, deterministic)
		s, r, dw, tr, info = envs.step(a)

		'''解析dones, wins, deads, truncateds, consistents信号：'''
		agent.queue.padding_with_done(~ct) # 根据上一时刻的ct去padding
		dones = envs.done_vec # (N)
		wins = envs.win_vec # (N)
		dead_and_tr = dones^wins # dones-wins = deads and truncateds
		ct = ~dones

		'''统计回合步数：'''
		step_collector += 1
		total_steps += step_collector[wins].sum() # 到达,总步数加上真实步数
		total_steps += (envs.max_ep_steps * dead_and_tr).sum() # 未到达,总步数加上回合最大步数
		step_collector[dones] = 0

		'''统计总奖励：'''
		r_collector += r
		total_r += r_collector[dones].sum()
		r_collector[dones] = 0

		'''统计到达率：'''
		arrived_vec += (~finished_vec & wins)  # 仅记录第一次win，防止二考刷分
		finished_vec += dones
		finished += dones.sum()

	return (total_steps.item() / finished.item(),
			total_r.item() / finished.item(),
			arrived_vec.sum().item() / opt.N)


def write(filename, data):
	# 打开文件以写入
	path = 'Evaluation_result'
	if not os.path.exists(path): os.mkdir(path)
	with open(path+'/'+filename, 'w') as f:
		# 定义列宽
		column_width = 15

		# 写入表头
		header = ['Index', 'Total Score', 'Arrival Score', 'Step Score', 'Reward Score']
		f.write('|'.join(f'{h:<{column_width}}' for h in header) + '\n')

		for item in data:
			# 使用相同的列宽写入数据
			line = '|'.join(f'{str(element):<{column_width}}' for element in item)
			f.write(line + '\n')  # 写入行并换行

if __name__ == '__main__':
	main()
