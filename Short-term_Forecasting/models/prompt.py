import torch
import torch.nn as nn
import numpy as np
from math import sqrt
# from kan import KANLinear
import torch
import torch.nn.functional as F
import math
# 原版代码，没有改动
# class Prompt(nn.Module):
#     def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
#                  prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform',wte = None):
#         super().__init__()
#         self.length = length
#         self.embed_dim = embed_dim
#         self.prompt_pool = prompt_pool
#         self.embedding_key = embedding_key
#         self.prompt_init = prompt_init
#         self.prompt_key = prompt_key
#         self.prompt_key_init = prompt_key_init
#         self.pool_size = pool_size
#         self.top_k = top_k
#         self.batchwise_prompt = batchwise_prompt
#         self.wte = wte
#         if self.prompt_pool:
#             prompt_pool_shape = (pool_size, length, embed_dim)
#             if prompt_init == 'zero':
#                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#             elif prompt_init == 'uniform':
#                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                 nn.init.uniform_(self.prompt, -1, 1)
#         # if using learnable prompt keys;
#         if prompt_key:
#             key_shape = (pool_size, embed_dim)
#             if prompt_key_init == 'zero':
#                 self.prompt = nn.Parameter(torch.zeros(key_shape),requires_grad=False)
#                 print('zero initialized key')
#             elif prompt_key_init == 'uniform':
#                 self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
#                 nn.init.uniform_(self.prompt, -5, 5)
#                 print('uniform initialized key')
#             elif prompt_key_init == 'gaussian':
#                 self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
#                 nn.init.normal_(self.prompt, mean=0.0, std=5.0)
#                 print('gaussian initialized key')
#             elif prompt_key_init == 'text_prototype':
#                 self.text_prototype_linear = nn.Linear(50257, pool_size)
#         else:
#             # else use mean of prompt as key
#             # only compatible with prompt, not prefix
#             prompt_mean = torch.mean(self.prompt, dim=1)
#             self.prompt_key = prompt_mean
#     def l2_normalize(self, x, dim=None, epsilon=1e-12):
#         """Normalizes a given vector or matrix."""
#         square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
#         x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
#         return x * x_inv_norm
#     def forward(self, x_embed, prompt_mask=None, cls_features=None):
#         out = dict()
#         if self.prompt_key:   #if self.prompt_pool:
#             if self.embedding_key == 'mean':
#                 x_embed_mean = torch.mean(x_embed, dim=1)
#             elif self.embedding_key == 'max':
#                 x_embed_mean = torch.max(x_embed, dim=1)[0]
#             elif self.embedding_key == 'mean_max':
#                 x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
#             elif self.embedding_key == 'cls':
#                 if cls_features is None:
#                     x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
#                 else:
#                     x_embed_mean = cls_features
#             else:
#                 raise NotImplementedError("Not supported way of calculating embedding keys!")
#             if self.prompt_key_init == 'text_prototype':
#                 prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
#             else:
#                 prompt_key = self.prompt
#             prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C   1000*768
#             x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C   512*768
#             similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
#             if prompt_mask is None:
#                 _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
#                 if self.batchwise_prompt:
#                     prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#                     # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#                     # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#                     # Unless dimension is specified, this will be flattend if it is not already 1D.
#                     if prompt_id.shape[0] < self.pool_size:
#                         prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                         id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#                     _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
#                     major_prompt_id = prompt_id[major_idx] # top_k
#                     # expand to batch
#                     idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
#             else:
#                 idx = prompt_mask # B, top_kdw
#             # batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
#             batched_prompt_raw = prompt_key[idx] # B, top_k, length, C
#             batched_prompt_raw = batched_prompt_raw.unsqueeze(2) # B, top_k, 1, length, C  1 1 1 768
#             batch_size, top_k, length, c = batched_prompt_raw.shape
#             batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C 1 1 768
#             out['prompt_idx'] = idx
#             # Debugging, return sim as well
#             out['prompt_norm'] = prompt_norm
#             out['x_embed_norm'] = x_embed_norm
#             out['similarity'] = similarity
#             # Put pull_constraint loss calculation inside
#             batched_key_norm = prompt_norm[idx] # B, top_k, C
#             out['selected_key'] = batched_key_norm
#             x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#             sim = batched_key_norm * x_embed_norm # B, top_k, C
#             reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
#             out['reduce_sim'] = reduce_sim
#         else:
#             if self.prompt_init == 'zero':
#                 self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
#             elif self.prompt_init == 'uniform':
#                 self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
#                 nn.init.uniform_(self.prompt)
#             batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
#         # The input with the prompt concatenated to the front. [B, prompt+token, C]
#         out['total_prompt_len'] = batched_prompt.shape[1]
#         out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
#         out['prompt_key'] = prompt_key  # prompt_key
#         return out
######################################################
# 以下是改动部分


# 稀疏注意力机制

############改动部分#
class Prompt(nn.Module):
    ####gpt2-medium 1024  gpt768
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform',wte = None):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.prob_attention = ProbAttention(mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                                            output_attention=False)
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.wte = wte
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        # if using learnable prompt keys;
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(key_shape),requires_grad=False)
                print('zero initialized key')
            elif prompt_key_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.uniform_(self.prompt, -5, 5)
                print('uniform initialized key')
            elif prompt_key_init == 'gaussian':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.normal_(self.prompt, mean=0.0, std=5.0)
                print('gaussian initialized key')
            elif prompt_key_init == 'text_prototype':
                #####MLP映射词汇表
                self.text_prototype_linear = nn.Linear(50257, pool_size)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_key:   #if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            if self.prompt_key_init == 'text_prototype':
                prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
            else:
                prompt_key = self.prompt
            prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C  1000,768
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C     512,768
            # similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size    1,1000

            #############改动部分#
            # # 计算 queries 和 keys
            # 假设 queries 的形状是 (L_Q, D)
            queries = x_embed_norm.unsqueeze(0).unsqueeze(2)   #变为 (B, 1, L_Q, 1)
            keys = prompt_norm.unsqueeze(0).unsqueeze(2)   # 变为 (B, 1, L_K, D)
            # 第一种使用 ProbAttention 计算稀疏注意力
            values = keys  # 可以使用 keys 作为 values
            attn_mask = None  # 如果不需要掩码，可以设置为 None
            context, _ = self.prob_attention(queries, keys, values, attn_mask)
            # context 的维度为 B, 1, C，去掉多余的维度
            similarity = context.squeeze(0)  # B, C     1 1 768
            #############第一种方式计算ProbAttention
        out['prompted_embedding'] = torch.concat((similarity,x_embed),dim=1)  #    # 1  66 768
        out['prompt_key'] = prompt_key  # prompt_key
        return out
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn








