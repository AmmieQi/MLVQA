#-----------------------------------------------
#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/26 20:31
#@Author  :Ma Jie
#@FileName: intra2inter.py
#-----------------------------------------------
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import numpy as np
np.set_printoptions(threshold=np.inf)

# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE/self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        print(att_map.cpu().numpy())

        test_map = torch.mean(att_map, dim=1)
        # print("-----------------------------------")
        # print(test_map.size())
        # data = test_map.cpu().numpy()
        # print(data[5, :14, :14])
        # print("-----------------------------------")

        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

# ---------------------
# ---- transformer ----
# ---------------------

class Transformer(nn.Module):
    def __init__(self, __C, layer):
        super(Transformer, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(layer)])

    def forward(self, ques, ques_mask):
        for enc in self.enc_list:
            ques = enc(ques, ques_mask)

        return ques

# ----------------------
# ---- position_enc ----
# ----------------------
class Position_enc(nn.Module):
    def __init__(self, __C):
        super(Position_enc, self).__init__()
        self.pos_embedding = nn.Embedding.from_pretrained(
            embeddings=self.get_sinusoid_encoding_table(14 + 1, __C.HIDDEN_SIZE, padding_idx=0),
            freeze=True
        )

    def forward(self, ques_ix):
        ques_len = torch.where(ques_ix > 0, torch.full_like(ques_ix, 1), ques_ix)
        ques_pos_enc = torch.arange(1, ques_ix.size()[1] + 1).repeat(ques_ix.size()[0], 1).cuda()

        # get the position of tokens
        ques_pos_enc = ques_pos_enc * ques_len
        ques_pos_enc = self.pos_embedding(ques_pos_enc)
        return ques_pos_enc

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)

# ----------------------------------
# ---- Inter-Modality Attention ----
# ----------------------------------
class IMA(nn.Module):
    def __init__(self, __C):
        super(IMA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, q, i, q_mask, i_mask):
        i = self.norm1(i + self.dropout1(
            self.mhatt(v=q, k=q, q=i, mask=q_mask)
        ))

        i = self.norm2(i + self.dropout2(
            self.ffn(i)
        ))

        return i



# -----------------------
# ---- Intra_2_inter ----
# -----------------------
class INTRA_2_INTER(nn.Module):
    def __init__(self, __C):
        super(INTRA_2_INTER, self).__init__()
        self.inter_att_list = nn.ModuleList(IMA(__C) for _ in range(__C.LAYER))

    def forward(self, ques, img, ques_mask, img_mask):
        for inter_att in self.inter_att_list:
            ques_update = inter_att(img, ques, img_mask, ques_mask)
            img_update  = inter_att(ques, img, ques_mask, img_mask)

            ques = ques_update
            img = img_update

