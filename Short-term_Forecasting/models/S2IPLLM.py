import numpy as np
import torch
import torch.nn as nn
from torch import optim

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
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from .prompt import Prompt
import pandas as pd
import math


# class S2IPLLM(nn.Module):
#
#     def __init__(self, configs, device):
#         super(S2IPLLM, self).__init__()
#         self.configs = configs
#         # self.is_gpt = configs.is_gpt
#         self.patch_size = configs.patch_size
#         # self.pretrain = configs.pretrain
#         self.stride = configs.stride
#         self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
#
#         self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
#         self.patch_num += 1
#
#         self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
#                                               output_hidden_states=True)  # loads a pretrained GPT-2 base model
#         # else:
#         #     print("------------------no pretrain------------------")
#         #     self.gpt2 = GPT2Model(GPT2Config())
#         self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
#         print("gpt2 = {}".format(self.gpt2))
#
#         self.in_layer = nn.Linear(configs.patch_size * 3, configs.d_model)
#
#         self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform',
#                                   prompt_pool=False,
#                                   prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length,
#                                   batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,
#                                   wte=self.gpt2.wte.weight)
#
#         self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num + configs.prompt_length)),
#                                    configs.pred_len)
#
#         # if configs.freeze and configs.pretrain:
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             if 'ln' in name or 'wpe' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#
#         self.cnt = 0
#
#     def forward(self, x, itr):
#         B, L, M = x.shape
#
#         means = x.mean(1, keepdim=True).detach()
#         x = x - means
#         stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
#         x /= stdev
#
#         x = rearrange(x, 'b l m -> (b m) l')
#
#         def decompose(x):
#             df = pd.DataFrame(x)
#             trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(
#                 method='ffill')
#             detrended = df - trend
#             seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(
#                 method='bfill').fillna(method='ffill')
#             residuals = df - trend - seasonal
#             combined = np.stack([trend, seasonal, residuals], axis=1)
#             return combined
#
#         decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
#         x = torch.tensor(decomp_results).to(self.gpt2.device)
#         x = rearrange(x, 'b l c d  -> b c (d l)', c=3)
#
#         x = self.padding_patch_layer(x)
#         x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#
#         x = rearrange(x, 'b c n p -> b n (c p)', c=3)
#         pre_prompted_embedding = self.in_layer(x.float())
#
#         outs = self.prompt_pool(pre_prompted_embedding)
#         prompted_embedding = outs['prompted_embedding']
#         sim = outs['similarity']
#         prompt_key = outs['prompt_key']
#         simlarity_loss = outs['reduce_sim']
#
#         last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
#         outputs = self.out_layer(last_embedding.reshape(B * M * 3, -1))
#
#         outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=3)
#         outputs = outputs.sum(dim=2)
#         outputs = rearrange(outputs, 'b m l -> b l m')
#
#         res = dict()
#         res['simlarity_loss'] = simlarity_loss
#
#         outputs = outputs * stdev
#         outputs = outputs + means
#
#         return outputs, res
##############改进后的模型
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
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
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
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

C = 3
class S2IPLLM(nn.Module):

    def __init__(self, configs, device):
        super(S2IPLLM, self).__init__()
        self.configs = configs
        # self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        # self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            # else:
            #     print("------------------no pretrain------------------")
            #     self.gpt2 = GPT2Model(GPT2Config())
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        print("gpt2 = {}".format(self.gpt2))



        self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)

        self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight)
        # self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)), configs.pred_len)
        self.out_layer = KANLinear(
            in_features=int(768 // C * ((self.patch_num + configs.prompt_length))),  ###patchnum 64
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


        # if configs.freeze and configs.pretrain:
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False



        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x = rearrange(x_enc, 'b l m -> (b m) l')

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
            if (C == 3):
                ###残差分解
                combined = np.stack([trend, seasonal, residuals], axis=1)
            elif (C == 1):
                ####无残差分解
                combined = np.stack([df], axis=1)
            return combined


        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())  # 不分解的结果32 512 3   分解的结果32
        # 改版代码
        # decomp_results = np.apply_along_axis(decompose, 1, x.detach().cpu().numpy())
        # decomp_results (512, 512, 3, 1)
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        # x.shape (512, 512, 3, 1)
        # x = x.unsqueeze(-1).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c=C)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        ###原文输入代码
        # x = rearrange(x, 'b c n p -> b n (c p)', c=C)       #x  224 64 48
        # pre_prompted_embedding = self.in_layer(x.float())  # 1 64 768
        # 通过 KANLinear 层，首先 reshape 为 [b*n, c*p]，输入到 KANLinear
        x = rearrange(x, 'b c n p -> b n (c p)', c=C)
        batch_size, seq_len, feature_size = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, feature_size)
        # 通过 waveKANLinear 层
        pre_prompted_embedding = self.in_layer(x_reshaped.float())
        # 恢复输出的形状为 [b, n, d_model]
        pre_prompted_embedding = pre_prompted_embedding.view(batch_size, seq_len, -1)
        # 原版代码
        outs = self.prompt_pool(pre_prompted_embedding)
        # 增加稀疏注意力机制
        # outs = self.sparse_attn(pre_prompted_embedding)nvi
        prompted_embedding = outs['prompted_embedding']  # 14 65 768  698880
        simlarity_loss = 0
        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        #####gpt2-medium 的降维操作（1024->768）
        # last_embedding = self.reduce_dim_layer(last_embedding)
        ######gpt2-medium 的降维操作（1024->768）
        outputs = self.out_layer(last_embedding.reshape(B * M * C, -1))
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=C)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')
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

        




        