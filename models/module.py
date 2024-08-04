import numpy as np
import math
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
import copy

class Transformer(nn.Module):

    def  __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # d_model = 32
        # dropout = 0.1
        # nhead = 8
        # dim_feed_forward = 128
        # num_encoder_layers = 2
        # num_decoder_layers = 2
        # normalize_before = False
        # return_intermediate_dec = True
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)  # layer, 6, norm

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.decoder_ = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model  # 256
        self.nhead = nhead  # 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # print('src.shape: {}'.format(src.shape))  # b 32 12 20
        src = src.flatten(2).permute(2, 0, 1)
        # print('src.shape: {}'.format(src.shape))  # 240(12*20) b 32

        # print('pos_embed.shape: {}'.format(pos_embed.shape))  # b 32 12 20
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # print('pos_embed.shape: {}'.format(pos_embed.shape))  # 240(12*20) b 32

        # print('query_embed.shape: {}'.format(query_embed.shape))  # 7 32
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # print('query_embed.shape: {}'.format(query_embed.shape))  # 7 b 32

        # print('mask.shape: {}'.format(mask.shape))  # 1 12 20
        mask = mask.flatten(1)
        # print('mask.shape: {}'.format(mask.shape))  # 1 240(12*20)

        tgt = torch.zeros_like(query_embed)
        # print('tgt.shape: {}'.format(tgt.shape))  # 7 b 32

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # print('memory.shape: {}'.format(memory.shape))  # 240(12*20) b 32

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # print('hs.shape: {}'.format(hs.shape))  # num_heads 7 b 32

        hs_ = self.decoder_(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return hs.transpose(1, 2), hs_.transpose(1, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # layer, 6
        self.num_layers = num_layers  # 6
        self.norm = norm  # layerNorm final norm?

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            # print('output.shape: {}'.format(output.shape))  # 1064 b 256

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)  # dec layer, 6
        self.num_layers = num_layers  # 6
        self.norm = norm  # layer norm
        self.return_intermediate = return_intermediate  # True

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src2, weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                                key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # print(nhead)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(hidden_dim,
                      dropout,
                      nheads,
                      dim_feedforward,
                      enc_layers,
                      dec_layers,
                      pre_norm=False,
                      return_intermediate_dec=False):

    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=return_intermediate_dec,
    )

def build_transformer_encoder(hidden_dim,
                              dropout,
                              n_heads,
                              dim_feedforward,
                              enc_layers,
                              pre_norm=False,):

    return TransformerEncoder(
        TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=pre_norm,
        ),
        enc_layers,
        nn.LayerNorm(hidden_dim) if pre_norm else None
    )

def build_transformer_decoder(hidden_dim,
                              dropout,
                              n_heads,
                              dim_feedforward,
                              dec_layers,
                              pre_norm=False,
                              return_intermediate=False):

    return TransformerDecoder(
        TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=pre_norm,
        ),
        dec_layers,
        nn.LayerNorm(hidden_dim),
        return_intermediate=return_intermediate
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x, mask=None):
        # x = tensor_list.tensors
        # mask = tensor_list.mask  # the image location which is padded with 0 is set to be 1 at the corresponding mask location
        # print(x.shape)  # b 128 8 8
        # print(mask.shape)  # b 8 8
        # exit()
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask  # image 0 -> 0 [B, H, W]

        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 2 28 38
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 2 28 38

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # [0~2pi]
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # [0~2pi]

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # [C]

        pos_x = x_embed[:, :, :, None] / dim_t   # [B, H, W, C//2]
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingSine3D(nn.Module):
    """
    This class extends positional embedding to 3d spaces.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, norm=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True
        self.norm = norm

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x_embed, y_embed, z_embed):
        if self.normalize:
            eps = 1e-6
            z_embed = (z_embed - self.norm[2]) / (self.norm[5] - self.norm[2] + eps) * self.scale
            y_embed = (y_embed - self.norm[1]) / (self.norm[4] - self.norm[1] + eps) * self.scale
            x_embed = (x_embed - self.norm[0]) / (self.norm[3] - self.norm[0] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x_embed.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t_z

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)  # [B, N, C//2]
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)  # [B, N, C]

        pos = torch.cat((pos_y, pos_x), dim=-1) + pos_z  # [B, N, C]

        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class AnchorGenerator(object):
    """Normalized anchor coords"""
    def __init__(self, anchor_cfg, y_steps=None, x_min=None, x_max=None, y_max=100, norm=None):
        self.y_steps = y_steps
        if self.y_steps is None:
            self.y_steps = np.linspace(1, y_max, y_max)
        self.pitches = anchor_cfg['pitches']
        self.yaws = anchor_cfg['yaws']
        self.num_x = anchor_cfg['num_x']
        self.anchor_len = len(self.y_steps)
        self.x_min = x_min
        self.x_max = x_max
        self.y_max = y_max
        self.norm = norm
        self.start_z = anchor_cfg.get('start_z', 0)

    def generate_anchors(self):
        anchors = []
        starts = [x for x in np.linspace(self.x_min, self.x_max, num=self.num_x, dtype=np.float32)]
        idx = 0
        for start_x in starts:
            for pitch in self.pitches:
                for yaw in self.yaws:
                    anchor = self.generate_anchor(start_x, pitch, yaw, start_z = self.start_z)
                    if anchor is not None:
                        anchors.append(anchor)
                        idx += 1
        self.anchor_num = len(anchors)
        print("anchor:", len(anchors))
        anchors = np.array(anchors)
        return anchors

    def generate_anchor(self, start_x, pitch, yaw, start_z=0, cut=True):
        # anchor [pos_score, neg_score, start_y, end_y, d, x_coords * 10, z_coords * 10, vis_coords * 10]
        anchor = np.zeros(2 + 2 + 1 + self.anchor_len * 3, dtype=np.float32)
        pitch = pitch * math.pi / 180.  # degrees to radians
        yaw = yaw * math.pi / 180.
        anchor[2] = 0
        anchor[3] = 1
        anchor[5:5+self.anchor_len] = start_x + self.y_steps * math.tan(yaw)
        anchor[5+self.anchor_len:5+self.anchor_len*2] = start_z + self.y_steps * math.tan(pitch)
        anchor_vis = np.logical_and(anchor[5:5+self.anchor_len] > self.x_min, anchor[5:5+self.anchor_len] < self.x_max)
        if cut:
            if sum(anchor_vis) / self.anchor_len < 0.5:
                return None
        return anchor
