import torch
from torch import nn

from models.common import Conv3D, Conv1D, FullyConnected, ESN
from utils.general import fix_problem_with_reuse_activation_funtion


class residual_block_1(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, act: any):
        super(residual_block_1, self).__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        self.m = nn.Sequential(Conv3D(in_channel, hidden_channel, kernel, 1, [None] * 3, act=act),
                               Conv3D(hidden_channel, hidden_channel, (1, 3, 3), 1, [None] * 3, act=act),
                               Conv3D(hidden_channel, out_channel, 1, 1, [None] * 3, act=False))
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        return self.act(inputs + self.m(inputs))


class residual_block_2(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, stride: any, act: any):
        super(residual_block_2, self).__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        self.m = nn.Sequential(Conv3D(in_channel, hidden_channel, kernel, 1, [None] * 3, act=act),
                               Conv3D(hidden_channel, hidden_channel, (1, 3, 3), stride, [None] * 3, act=act),
                               Conv3D(hidden_channel, out_channel, 1, 1, [None] * 3, act=False))
        self.m1 = Conv3D(in_channel, out_channel, 1, stride, [None] * 3, act=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        return self.act(self.m1(inputs) + self.m(inputs))


class SlowFast(nn.Module):
    export = False

    def __init__(self, in_channel, time_keeper=4, nc=1000, use_liquid=False, act=nn.ReLU()):
        super(SlowFast, self).__init__()
        self.time_keeper = time_keeper
        slow_output_channel = 64
        fast_to_slow_output_channel = 16
        fast_output_channel = 8
        self.slow_branch_conv0 = nn.Sequential(
            Conv3D(in_channel, slow_output_channel, (1, 7, 7), (1, 2, 2), [None] * 3, act=act),
            nn.MaxPool3d((1, 7, 7), (1, 2, 2), (0, 3, 3)))
        self.fast_branch_conv0 = nn.Sequential(
            Conv3D(in_channel, fast_output_channel, (5, 7, 7), (1, 2, 2), [None] * 3, act=act),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))

        self.conv_fast_to_slow0 = Conv3D(8, fast_to_slow_output_channel, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        slow_output_channel = slow_output_channel + fast_to_slow_output_channel
        self.slow_branch_res0 = residual_block_2(slow_output_channel, 256, 64, [1] * 3, [1] * 3, act=act)
        self.fast_branch_res0 = residual_block_2(fast_output_channel, 32, fast_output_channel, (3, 1, 1), [1] * 3,
                                                 act=act)
        slow_branch_res1 = [residual_block_1(256, 256, 64, [1] * 3, act=act)] * 2
        self.slow_branch_res1 = nn.Sequential(*slow_branch_res1)
        fast_branch_res1 = [residual_block_1(32, 32, 8, (3, 1, 1), act=act)] * 2
        self.fast_branch_res1 = nn.Sequential(*fast_branch_res1)
        # ---
        self.conv_fast_to_slow1 = Conv3D(32, 64, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        self.slow_branch_res2 = residual_block_2(320, 512, 128, [1] * 3, (1, 2, 2), act=act)
        self.fast_branch_res2 = residual_block_2(32, 64, 16, (3, 1, 1), (1, 2, 2), act=act)
        slow_branch_res3 = [residual_block_1(512, 512, 128, [1] * 3, act=act)] * 3
        self.slow_branch_res3 = nn.Sequential(*slow_branch_res3)
        slow_branch_res4 = [residual_block_1(64, 64, 16, (3, 1, 1), act=act)] * 3
        self.fast_branch_res3 = nn.Sequential(*slow_branch_res4)

        # ---
        self.conv_fast_to_slow2 = Conv3D(64, 128, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        self.slow_branch_res4 = residual_block_2(640, 1024, 256, (3, 1, 1), (1, 2, 2), act=act)
        self.fast_branch_res4 = residual_block_2(64, 128, 32, (3, 1, 1), (1, 2, 2), act=act)
        slow_branch_res5 = [residual_block_1(1024, 1024, 256, (3, 1, 1), act=act)] * 5
        self.slow_branch_res5 = nn.Sequential(*slow_branch_res5)
        fast_branch_res5 = [residual_block_1(128, 128, 32, (3, 1, 1), act=act)] * 5
        self.fast_branch_res5 = nn.Sequential(*fast_branch_res5)
        # ---
        self.conv_fast_to_slow3 = Conv3D(128, 256, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        self.slow_branch_res6 = residual_block_2(1280, 2048, 512, [1] * 3, (1, 2, 2), act=act)
        self.fast_branch_res6 = residual_block_2(128, 256, 64, (3, 1, 1), (1, 2, 2), act=act)
        slow_branch_res7 = [residual_block_1(2048, 2048, 512, (3, 1, 1), act=act)] * 2
        self.slow_branch_res7 = nn.Sequential(*slow_branch_res7)
        fast_branch_res7 = [residual_block_1(256, 256, 64, (3, 1, 1), act=act)] * 2
        self.fast_branch_res7 = nn.Sequential(*fast_branch_res7)
        if use_liquid:
            self.fully = ESN(2304, 2500, nc)
        else:
            self.conv = Conv1D(2304, 2480, act=act)
            self.fully = FullyConnected(2480, nc, act=False)
        self.pool = nn.AdaptiveAvgPool3d(4)
        self.post_act = nn.Softmax(1)
        self.use_liquid = use_liquid

    def forward(self, inputs):
        slow_branch, fast_branch = inputs[:, :, ::self.time_keeper, ...], inputs
        slow_branch, fast_branch = self.slow_branch_conv0(slow_branch), self.fast_branch_conv0(fast_branch)
        slow_branch = torch.cat([slow_branch, self.conv_fast_to_slow0(fast_branch)], dim=1)
        # ---
        slow_branch = self.slow_branch_res0(slow_branch)
        fast_branch = self.fast_branch_res0(fast_branch)
        slow_branch = self.slow_branch_res1(slow_branch)
        fast_branch = self.fast_branch_res1(fast_branch)
        # ---
        slow_branch = torch.cat([self.conv_fast_to_slow1(fast_branch), slow_branch], dim=1)
        # # --
        slow_branch = self.slow_branch_res2(slow_branch)
        fast_branch = self.fast_branch_res2(fast_branch)
        slow_branch = self.slow_branch_res3(slow_branch)
        fast_branch = self.fast_branch_res3(fast_branch)
        # ---
        slow_branch = torch.cat([self.conv_fast_to_slow2(fast_branch), slow_branch], dim=1)
        # ---
        slow_branch = self.slow_branch_res4(slow_branch)
        fast_branch = self.fast_branch_res4(fast_branch)
        slow_branch = self.slow_branch_res5(slow_branch)
        fast_branch = self.fast_branch_res5(fast_branch)
        # ---
        slow_branch = torch.cat([self.conv_fast_to_slow3(fast_branch), slow_branch], dim=1)
        # ---
        slow_branch = self.slow_branch_res6(slow_branch)
        fast_branch = self.fast_branch_res6(fast_branch)
        slow_branch = self.slow_branch_res7(slow_branch)
        fast_branch = self.fast_branch_res7(fast_branch)

        slow_branch = self.pool(slow_branch)
        fast_branch = self.pool(fast_branch)
        outputs = torch.cat([slow_branch, fast_branch], dim=1)
        b, c, d, h, w = outputs.shape
        outputs = outputs.view(-1, c, d * h * w)
        if not self.use_liquid:
            outputs = self.conv(outputs)
        outputs = outputs.permute((0, 2, 1))
        if not self.use_liquid:
            outputs = outputs.mean(1)
        outputs = self.fully(outputs)

        if torch.onnx.is_in_onnx_export() or self.export:
            outputs = self.post_act(outputs)

        return outputs
