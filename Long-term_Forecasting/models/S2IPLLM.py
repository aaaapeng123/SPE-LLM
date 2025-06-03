#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
# from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr
from .prompt import Prompt
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3, # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。
        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # print(f"shape of input: {x.shape}")
        # print(f"in_features: {self.in_features}")
        # print(f"x.size(1): {x.size(1)}")
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
C = 3
############趋势分解数量
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        # self.mix_channel = ChannelMixMLP(7, 7)
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.reduce_dim_layer = nn.Linear(1024, 768)
        self.d_ff = 768####gpt2-medium 1024  gpt768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.fan_model = FAN(seq_len=self.seq_len, pred_len=self.seq_len, enc_in=7, freq_topk=20, rfft=True)
        if configs.pretrained == True:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False
        if self.task_name == 'long_term_forecast':
            #做了C个分解之后
            self.in_layer = nn.Linear(configs.patch_size * C, configs.d_model)
            ####小波变换kanin_layer
            # self.in_layer = WaveKANLinear(
            #     in_features=configs.patch_size * C,
            #     out_features=configs.d_model,
            #     wavelet_type='mexican_hat'  # 可以选择 'morlet', 'dog', 'meyer', 'shannon' 等
            # )
            #####原版代码out_layer
            # self.out_layer = nn.Linear(int(768 // C * ((self.patch_num + configs.prompt_length))),
            #                            configs.pred_len)
            #######KAN模型out_layer
            self.out_layer = KANLinear(
                in_features=int(768 // C * ((self.patch_num + configs.prompt_length))),###patchnum 64
                out_features=configs.pred_len,
                grid_size=7,  # 你可以根据需要调整这些参数
                spline_order=3,
                scale_noise=0.05,
                scale_base=1.0,
                scale_spline=1.0,
                enable_standalone_scale_spline=True,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1]
            )
            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                                      prompt_pool=False,
                                      prompt_key=True, pool_size=self.configs.pool_size,
                                      top_k=self.configs.prompt_length, batchwise_prompt=False,
                                      prompt_key_init=self.configs.prompt_init, wte=self.gpt2.wte.weight)
            for layer in (self.gpt2, self.in_layer, self.out_layer):
                layer.cuda()
                layer.train()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out, res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]
        return None
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape #32 512 7
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x = rearrange(x_enc, 'b l m -> (b m) l')  #224 512
        # x [512,512]
        # 原版计算趋势和季节性的函数
        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(
                method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(
                method='bfill').fillna(method='ffill')

            residuals = df - trend - seasonal
            if(C==3):
                ###残差分解
                combined = np.stack([trend, seasonal, residuals], axis=1)
            elif(C==1):
                ####无残差分解
                combined = np.stack([df], axis=1)
            return combined

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())  # 224 512 3 1
        # 改版代码
        # decomp_results = np.apply_along_axis(decompose, 1, x.detach().cpu().numpy())
        # decomp_results (512, 512, 3, 1)
        x = torch.tensor(decomp_results).to(self.gpt2.device)# 224 512 3 1
        # x.shape (512, 512, 3, 1)
        # x = x.unsqueeze(-1).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c=C)#224 3 512
        x = self.padding_patch_layer(x)  # 224 3 520
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # 224 3 64 16
        ###原文输入代码
        # x = rearrange(x, 'b c n p -> b n (c p)', c=C)       #x  224 64 48
        # pre_prompted_embedding = self.in_layer(x.float())  # 1 64 768
        # 通过 KANLinear 层，首先 reshape 为 [b*n, c*p]，输入到 KANLinear
        x = rearrange(x, 'b c n p -> b n (c p)', c=C)  # x  224 64 48
        batch_size, seq_len, feature_size = x.size()
        x_reshaped = x.view(batch_size * seq_len, feature_size)
        # 通过 waveKANLinear 层
        pre_prompted_embedding = self.in_layer(x_reshaped.float())
        # 恢复输出的形状为 [b, n, d_model]
        pre_prompted_embedding = pre_prompted_embedding.view(batch_size, seq_len, -1)  # 224 64 768
        # 原版代码
        outs = self.prompt_pool(pre_prompted_embedding)
        # 增加稀疏注意力机制
        # outs = self.sparse_attn(pre_prompted_embedding)nvi
        prompted_embedding = outs['prompted_embedding']  # 224 65 768           698880
        simlarity_loss = 0
        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state  # 224 65 768
        outputs = self.out_layer(last_embedding.reshape(B * M * C, -1))# 672 96
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=C) # 32 7 3 96
        outputs = outputs.sum(dim=2)    # 32 7 96
        outputs = rearrange(outputs, 'b m l -> b l m')   # 32 96 7
        res = dict()
        res['simlarity_loss'] = simlarity_loss
        # 增加循环周期
        # outputs = outputs + self.cycleQueue((96 + self.seq_len) % 96, self.pred_len)
        ########### #分布式修改部分
        stdev = stdev.to(outputs.device)
        means = means.to(outputs.device)
        ############分布式修改部分
        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]
        # outputs [512,96,1]
        return outputs, res
