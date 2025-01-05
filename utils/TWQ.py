import torch


class TimeWindowQueue_NTD:
    def __init__(self, N, T, D, device, padding):
        self.device = device
        self.padding = padding
        self.T = T

        # 初始化缓冲区，形状为 (N, T, D), 即transformer(batch_first=True)要求的(batch_size, seq_len, emb_dim)
        if padding == -1: self.window = -torch.ones(N, T, D, device=self.device) # 便于可视化
        elif padding == 0: self.window = torch.zeros(N, T, D, device=self.device)
        else: raise ValueError("Wrong padding value")
        self.ptr = 0


    def append(self, batched_transition: torch.tensor):
        """ batched_transition, shape=(B,D): batched transition from vectorized envs """

        # 将数据写入缓冲区
        self.window[:, self.T - 1 - self.ptr, :] = batched_transition # (B,D), 由下往上写入，保证roll的输出顺序

        # 更新写指针和计数器
        self.ptr = (self.ptr + 1) % self.T

    def get(self) -> torch.tensor:
        """
        获取时间窗口buffer中的所有数据, shape=(N, T, D), 使用roll保证数据按时序正确排列
        t=0为最近时刻的数据, t=T-1为最远时刻的数据
        """
        TimeWindow_data = torch.roll(self.window, shifts=self.ptr, dims=1) # (N, T, D)

        return TimeWindow_data


    def padding_with_done(self, done_flag: torch.tensor):
        """
        根据done_flag，将buffer中对应batch位置置零
        :param done_flag: shape=(N,)
        """
        self.window[done_flag, :, :] = self.padding

    def clear(self):
        self.window.fill_(self.padding)


