from multiprocessing.managers import BaseManager
import torch.multiprocessing as mp
from ASL.actor import actor_process
from ASL.sharer import shared_data
from ASL.learner import learner_process
from Sparrow_V2 import str2bool
import argparse
import torch


if __name__ == '__main__':
    '''Hyperparameter Setting for DRL'''
    parser = argparse.ArgumentParser()
    # running devices configuration
    parser.add_argument('--Env_dvc', type=str, default='cuda:0', help='running device for Sparrow Env')
    parser.add_argument('--A_dvc', type=str, default='cuda:0', help='running device for Actor')
    parser.add_argument('--B_dvc', type=str, default='cuda:0', help='running device for Buffer of Sharer')
    parser.add_argument('--L_dvc', type=str, default='cuda:0', help='running device for Learner')

    # training strategy configuration
    parser.add_argument('--exp_name', type=str, default='ColorDynamic', help='Experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_train_steps', type=int, default=int(3E6), help='Max training total steps')
    parser.add_argument('--explore_steps', type=int, default=int(3E4), help='Random warm up total steps before training.')
    parser.add_argument('--TPS', type=int, default=256, help='Transitions been trained Per single Step')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--time_feedback', type=str2bool, default=True, help='Whether use time feedback mechanism')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--clip', type=int, default=10, help='clip_grad_norm for model optimization')
    parser.add_argument('--soft_target', type=str2bool, default=False, help='Target net update mechanism')
    parser.add_argument('--save_freq', type=int, default=int(5e4), help='Model save frequency, in Bstep')
    parser.add_argument('--reset_freq', type=int, default=int(32E3), help='training env reset frequency (curriculum learning), in total steps')
    parser.add_argument('--upload_freq', type=int, default=int(500), help='learner update freq, in Bstep')

    # vectorized e-greedy configuration
    parser.add_argument('--init_explore_frac', type=float, default=1.0, help='init explore fraction')
    parser.add_argument('--end_explore_frac', type=float, default=0.2, help='end explore fraction')
    parser.add_argument('--decay_step', type=int, default=int(100E3), help='linear decay steps(total) for e-greedy noise')
    parser.add_argument('--min_eps', type=float, default=0.05, help='minimal e-greedy noise')

    # Transqer configuration
    parser.add_argument('--net_width', type=int, default=64, help='Linear net width')
    parser.add_argument('--T', type=int, default=10, help='length of time window')
    parser.add_argument('--H', type=int, default=8, help='Number of Head')
    parser.add_argument('--L', type=int, default=3, help='Number of Transformer Encoder Layers')

    '''Hyperparameter Setting for Sparrow'''
    parser.add_argument('--dvc', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
    parser.add_argument('--action_type', type=str, default='Discrete', help='Action type: Discrete / Continuous')
    parser.add_argument('--window_size', type=int, default=800, help='size of the training map')
    parser.add_argument('--D', type=int, default=400, help='maximal local planning distance')
    parser.add_argument('--N', type=int, default=32, help='number of vectorized environments')
    parser.add_argument('--O', type=int, default=15, help='number of obstacles in each environment')
    parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
    parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
    parser.add_argument('--RdOV', type=str2bool, default=False, help='whether to randomize the Velocity of dynamic obstacles')
    parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
    parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
    parser.add_argument('--Obs_R', type=int, default=14, help='maximal obstacle radius, cm')
    parser.add_argument('--Obs_V', type=int, default=50, help='maximal obstacle velocity, cm/s')
    parser.add_argument('--MapObs', type=str, default=None, help="name of map file, e.g. 'map.png' or None")
    parser.add_argument('--ld_a_range', type=int, default=360, help='max scanning angle of lidar (degree)')
    parser.add_argument('--ld_d_range', type=int, default=300, help='max scanning distance of lidar (cm)')
    parser.add_argument('--ld_num', type=int, default=72, help='number of lidar streams in each world')
    parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for each group')
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
    opt.render_mode = None # dont render when training
    opt.buffersize = min(int(1E6), opt.max_train_steps)
    opt.reset_freq = int(opt.reset_freq / opt.N)  # Tsteps -> Vsteps

    opt.dvc = torch.device(opt.Env_dvc)
    opt.state_dim = 8+int(opt.ld_num/opt.ld_GN)
    opt.action_dim = 7

    # Set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # 注册一个类，凡是注册到管理器中的类/对象，都可以被不同进程共享,"shared_data"是注册的名字，后面实例化时要调用这个名字
    BaseManager.register("shared_data", callable=shared_data)
    ShareManager = BaseManager()
    ShareManager.start()
    opt.shared_data = ShareManager.shared_data(opt)  # 实例化一个shared_data

    processes = []
    # actor process
    processes.append(mp.Process(target=actor_process, args=(opt, )))
    processes[-1].start()

    # learner process
    processes.append(mp.Process(target=learner_process, args=(opt, )))
    processes[-1].start()


    for _ in range(len(processes)):
        processes[_].join()








