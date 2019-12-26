#-----------------------------------------------
#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/26 20:32
#@Author  :Ma Jie
#@FileName: net.py
#-----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from openvqa.models.mlvqa.adapter import Adapter
from openvqa.models.mlvqa.intra2inter import INTRA_2_INTER
from openvqa.models.mlvqa.intra2inter import Position_enc, Transformer
from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import MLP
from openvqa.ops.layer_norm import LayerNorm

class Flat_vec(nn.Module):
    def __init__(self, __C):
        super(Flat_vec, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # project 300-dimensional word embedding to 512-dimensional
        self.proj = nn.Linear(__C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE)

        # position encoding
        self.pos_enc = Position_enc(__C)

        # self.lstm = nn.LSTM(
        #     input_size=__C.WORD_EMBED_SIZE,
        #     hidden_size=__C.HIDDEN_SIZE,
        #     num_layers=1,
        #     batch_first=True
        # )
        self.ques_intra = Transformer(__C, __C.QUES_LAYER)

        self.adapter = Adapter(__C)

        # image encoding (intra-modality attention of image)
        self.img_intra = Transformer(__C, __C.IMG_LAYER)
        self.backbone = INTRA_2_INTER(__C)

        # use it to project vector into the space of question vocab.
        self.proj_norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.classifier1 = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_OUT_SIZE,
            out_size=token_size,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

       # flatten vector and project it into answer space
        self.flat_img = Flat_vec(__C)
        self.flat_lang = Flat_vec(__C)

        # use it to project the vector to answer space.
        self.proj_norm2 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.classifier2 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix_no_msk, ques_ix):
        #======================================================================#
        #---------use the masked strategy to predict the masked token----------#
        #======================================================================#

        # pre-process question/language feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)

        lang_feat = self.proj(lang_feat)
        pos_feat = self.pos_enc(ques_ix)

        # lang_feat, _ = self.lstm(lang_feat)
        lang_feat = self.ques_intra(lang_feat + pos_feat, lang_feat_mask)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        img_feat = self.img_intra(img_feat, img_feat_mask)
        img_feat_new = img_feat

        # backbone framework (intra-self attention -> inter-modality attention)
        lang_feat_fusion, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # predict the masked token
        lang_feat_fusion = lang_feat_fusion + lang_feat
        lang_feat_fusion = self.proj_norm1(lang_feat_fusion)
        proj_feat1 = self.classifier1(lang_feat_fusion)

        # ======================================================================#
        # -use the original information (no mask) to predict the answer of vqa--#
        # ======================================================================#

        lang_feat_original_msk = make_mask(ques_ix_no_msk.unsqueeze(2))
        lang_feat_original = self.embedding(ques_ix_no_msk)

        lang_feat_original = self.proj(lang_feat_original)
        pos_feat_original = self.pos_enc(ques_ix_no_msk)

        lang_feat_original = self.ques_intra(lang_feat_original + pos_feat_original, lang_feat_original_msk)

        # lang_feat_original, _ = self.lstm(lang_feat_original)
        # lang_feat_original = self.ques_intra(lang_feat_original, lang_feat_original_msk)

        # img_feat_new, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        # img_feat_new = self.img_intra(img_feat, img_feat_mask)

        lang_feat_original, img_feat_new = self.backbone(
            lang_feat_original,
            img_feat_new,
            lang_feat_original_msk,
            img_feat_mask
        )

        # flatten vector
        lang_feat_original = self.flat_lang(
            lang_feat_original,
            lang_feat_original_msk
        )
        img_feat_new = self.flat_img(
            img_feat_new,
            img_feat_mask
        )

        # predict the vqa answer
        proj_feat2 = lang_feat_original + img_feat_new
        proj_feat2 = self.proj_norm2(proj_feat2)
        proj_feat2 = self.classifier2(proj_feat2)

        return proj_feat1, proj_feat2