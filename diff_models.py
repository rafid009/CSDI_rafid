import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
import math
from pypots.imputation.transformer import EncoderLayer, PositionalEncoding
from pypots.imputation import SAITS
import numpy as np
# torch.manual_seed(42)

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv1d_with_init_saits(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


############################### New Design ################################

# def swish(x):
#     return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        # self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    

def Conv1d_with_init_saits_new(in_channels, out_channels, kernel_size, init_zero=False):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # layer = nn.utils.weight_norm(layer)
    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer
    
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out

def get_output_size(H_in, K, s):
    return int((H_in - K)/s) + 1

def conv_with_init(in_channels, out_channel, kernel_size):
    layer = nn.Conv2d(in_channels, out_channel, kernel_size, stride=2)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class ResidualEncoderLayer_2(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()


        # before combi 2
        # self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
        #                  diagonal_attention_mask)
        # combi 2
        self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        
        self.enc_layer_2 = EncoderLayer(d_time, actual_d_feature, 2 * channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)

        # self.enc_layer_f = EncoderLayer(channels, d_time, d_time, d_inner, n_head, d_k, d_v, dropout, 0,
        #                  diagonal_attention_mask)

        # self.init_projection = Conv1d_with_init(2, channels, 1)
        # self.mid_projection = Conv1d_with_init(int(channels / 2), 2 * channels, 1)
        # self.output_projection = Conv1d_with_init(channels, 4, 1)
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # self.out_skip_proj = Conv1d_with_init(2, 1, 1)
        # self.pre_mid_proj = Conv1d_with_init(channels, int(channels / 2), 1)

        self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        # before combi 2
        # self.conv_layer = Conv1d_with_init_saits_new(channels, channels, kernel_size=1)
        # cmobi 2
        self.conv_layer = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.cond_proj = Conv1d_with_init_saits_new(d_model, channels, 1)

        # before combi 2
        # self.conv_noisy = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.conv_cond = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)


        self.res_proj = Conv1d_with_init_saits_new(channels, d_model, 1)
        self.skip_proj = Conv1d_with_init_saits_new(channels, d_model, 1)

        # self.output_proj = Conv1d_with_init_saits_new(channels, 2 * d_model, 1)
        
        # self.norm = nn.LayerNorm([d_time, d_model])
        # self.post_enc_proj = Conv1d_with_init(channels, 4, 1)



    # new_design
    def forward(self, x, cond, diffusion_emb):
        # x Noise
        # L -> feature
        # K -> time
        B, K, L = x.shape

        x_proj = torch.transpose(x, 1, 2) # (B, L, K)
        x_proj = self.init_proj(x_proj)

        cond = torch.transpose(cond, 1, 2) # (B, L, K)
        cond = self.cond_proj(cond)
        

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_proj + diff_proj + cond


        y = torch.transpose(y, 1, 2) # (B, K, channels)
        y, _ = self.enc_layer_1(y)
        y = torch.transpose(y, 1, 2)


        y = self.conv_layer(y)
        c_y = self.conv_cond(cond)
        y = y + c_y


        y = torch.transpose(y, 1, 2) # (B, K, 2*channels)
        y, attn_weights_2 = self.enc_layer_2(y)
        y = torch.transpose(y, 1, 2)
 

        y1, y2 = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(y1) * torch.tanh(y2) # (B, channels, K)

        residual = self.res_proj(out) # (B, L, K)
        residual = torch.transpose(residual, 1, 2) # (B, K, L)

        skip = self.skip_proj(out) # (B, L, K)
        skip = torch.transpose(skip, 1, 2) # (B, K, L)


        attn_weights = attn_weights_2 #torch.softmax(attn_weights_1 + attn_weights_2, dim=-1)

        return (x + residual) * math.sqrt(0.5), skip, attn_weights


class diff_SAITS_3(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        self.d_feature = d_feature
        channels = d_model #int(d_model / 2)
        
        self.layer_stack_for_first_block = nn.ModuleList([
            ResidualEncoderLayer_2(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                        d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                        diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        self.layer_stack_for_second_block = nn.ModuleList([
            ResidualEncoderLayer_2(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                        d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                        diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.position_enc_cond = PositionalEncoding(d_model, n_position=d_time)
        self.position_enc_noise = PositionalEncoding(d_model, n_position=d_time)

        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.embedding_cond = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_skip_z = nn.Linear(d_model, d_feature)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

        # combi 2: trying feature weights here
        # self.feature_weights = EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_k, d_v, dropout, 0,
        #                  True)
        
        # combi 2 more layers
        self.layer_stack_for_feature_weights = nn.ModuleList([
            EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_k, d_v, dropout, 0,
                         True)
            for _ in range(n_layers)
        ])

        

    # ds3
    def forward(self, inputs, diffusion_step):
        # print(f"Entered forward")
        X, masks = inputs['X'], inputs['missing_mask']
        
        ## making the mask same

        masks[:,1,:,:] = masks[:,0,:,:]
        # B, K, L -> B=batch, K=time, L=feature
        X = torch.transpose(X, 2, 3)
        masks = torch.transpose(masks, 2, 3)

        # combi 2:
        cond_X = X[:,0,:,:] + X[:,1,:,:]
        cond_X = torch.transpose(cond_X, 1, 2)
        # only combi 2
        # cond_X, attn_weights_f = self.feature_weights(cond_X)
        # combi 2 with more layers
        for feat_enc_layer in self.layer_stack_for_feature_weights:
            cond_X, attn_weights_f = feat_enc_layer(cond_X)
        cond_X = torch.transpose(cond_X, 1, 2)
        

        # before combi 2
        # input_X_for_first = torch.cat([X[:,1,:,:], masks[:,1,:,:]], dim=2)
        # input_X_for_first = self.embedding_1(input_X_for_first)

        # combi 2
        input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)


        # cond separate
        noise = input_X_for_first
        cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
        cond = self.embedding_cond(cond)

        diff_emb = self.diffusion_embedding(diffusion_step)
        pos_cond = self.position_enc_cond(cond)

        
        enc_output = self.dropout(self.position_enc_noise(noise))
        skips_tilde_1 = torch.zeros_like(enc_output)
        for encoder_layer in self.layer_stack_for_first_block:
            # new time add
            enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
            # old stable better
            # enc_output, skip, _ = encoder_layer(enc_output, pos_cond, diff_emb)
            skips_tilde_1 += skip
        skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
        skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)
        
        # new time add
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)
            attn_weights_f = torch.softmax(attn_weights_f, dim=-1)

        # Feature attention added
        attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights_f.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
            attn_weights_f = attn_weights_f.mean(dim=3) 
            attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
            attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
        X_tilde_1 = self.reduce_dim_z(enc_output)
        # Old stable better
        X_tilde_1 = X_tilde_1 @ attn_weights_f + X_tilde_1
        # new time add
        X_tilde_1 = torch.transpose(X_tilde_1, 1, 2)
        X_tilde_1 = X_tilde_1 @ attn_weights + X_tilde_1
        X_tilde_1 = torch.transpose(X_tilde_1, 1, 2)

        # Old stable better
        # X_tilde_1 = X_tilde_1 + X[:, 1, :, :] 

        # second DMSA block

        # before combi 2
        input_X_for_second = torch.cat([X_tilde_1, masks[:,1,:,:]], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        noise = input_X_for_second

        enc_output = self.position_enc_noise(noise)
        skips_tilde_2 = torch.zeros_like(enc_output)
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
            skips_tilde_2 += skip


        # skip_tilde_2
        skips_tilde_2 /= math.sqrt(len(self.layer_stack_for_second_block))
        skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
        )  # namely term eta

        skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2

        skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
        skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
        skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)

        return skips_tilde_1, skips_tilde_2, skips_tilde_3
    




class diff_SAITS_4(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        self.d_feature = d_feature
        channels = d_model #int(d_model / 2)
        
        self.layer_stack_for_first_block = nn.ModuleList([
            ResidualEncoderLayer_3(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                        d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                        diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        # self.layer_stack_for_second_block = nn.ModuleList([
        #     ResidualEncoderLayer_2(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
        #                 d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
        #                 diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
        #     for _ in range(n_layers)
        # ])
        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.position_enc_cond = PositionalEncoding(d_model, n_position=d_time)
        self.position_enc_noise = PositionalEncoding(d_model, n_position=d_time)

        # self.conv_cond_X = Conv1d_with_init_saits_new(2, 1, 1)

        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.embedding_cond = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        # self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_skip_z = nn.Linear(d_model, d_feature)
        # self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

        # combi 2: trying feature weights here
        # self.feature_weights = EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_k, d_v, dropout, 0,
        #                  True)
        
        # combi 2 more layers
        self.layer_stack_for_feature_weights = nn.ModuleList([
            EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_k, d_v, dropout, 0,
                         True)
            for _ in range(n_layers)
        ])

        

    # ds4
    def forward(self, inputs, diffusion_step):
        # print(f"Entered forward")
        X, masks = inputs['X'], inputs['missing_mask']
        shape = X[:,0,:,:].shape # (B, L, K)
        ## making the mask same

        masks[:,1,:,:] = masks[:,0,:,:]
        # B, K, L -> B=batch, K=time, L=feature
        X = torch.transpose(X, 2, 3)
        masks = torch.transpose(masks, 2, 3)


        # cond separate
        cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
        cond = self.embedding_cond(cond)
        diff_emb = self.diffusion_embedding(diffusion_step)
        pos_cond = self.position_enc_cond(cond)

        cond_X = X[:,1,:,:] # (B, K, L)
        skips = []
        combine_weight = None
        K = 2
        for k in range(K):
            skips_tilde_1 = torch.zeros_like(cond)
            for i in range(len(self.layer_stack_for_first_block)):
                # cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, K, L)
                # cond_X = torch.transpose(cond_X, 2, 3) # (B, 2, L, K)
                # cond_X = cond_X.reshape(shape[0], -1, shape[1] * shape[2]) # (B, 2, L*K)
                # cond_X = self.conv_cond_X(cond_X) # (B, 1, L*K)
                # cond_X = torch.squeeze(cond_X, dim=1)
                # cond_X = cond_X.reshape(shape[0], shape[1], shape[2]) # (B, L, K)
                if i == 0:
                    cond_X = cond_X + X[:,0,:,:]
                cond_X = torch.transpose(cond_X, 1, 2)
                cond_X, attn_weights_f = self.layer_stack_for_feature_weights[i](cond_X)
                cond_X = torch.transpose(cond_X, 1, 2)

                input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
                noise = self.embedding_1(input_X_for_first)
                
                if i == 0 and k == 0:
                    enc_output = self.dropout(self.position_enc_noise(noise))
                else:
                    enc_output = self.position_enc_noise(noise)

                enc_output, skip, attn_weights = self.layer_stack_for_first_block[i](enc_output, pos_cond, diff_emb)

                # Time attentions
                attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights = torch.transpose(attn_weights, 1, 3)
                    attn_weights = attn_weights.mean(dim=3)
                    attn_weights = torch.transpose(attn_weights, 1, 2)
                    attn_weights = torch.softmax(attn_weights, dim=-1)

                if k == K - 1 and i == len(self.layer_stack_for_first_block) - 1:
                    combine_weight = attn_weights

                enc_output = torch.transpose(enc_output, 1, 2) # (B, L, K)
                enc_output = enc_output @ attn_weights 
                enc_output = torch.transpose(enc_output, 1, 2) # (B, K, L)
                X_tilde_1 = self.reduce_dim_z(enc_output)
                # Feature attention
                attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights_f.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
                    attn_weights_f = attn_weights_f.mean(dim=3) 
                    attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
                    attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
                
                # if (i + 1) % 2 == 0:
                cond_X = X_tilde_1 @ attn_weights_f + X_tilde_1 + X[:, 1, :, :]
                # else:
                #     cond_X = X_tilde_1 @ attn_weights_f + X_tilde_1 #+ X[:, 1, :, :]
                skips_tilde_1 += skip
            skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
            if k == 0:
                skips_tilde_1 = (self.reduce_skip_z(skips_tilde_1))
            else:
                skips_tilde_1 = self.reduce_dim_gamma(F.relu(self.reduce_skip_z(skips_tilde_1)))
            skips.append(skips_tilde_1)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks[:, 0, :, :], combine_weight], dim=2))
        )
        skips_tilde_1 = skips[0]
        skips_tilde_2 = skips[1]
        skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2

        skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
        skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
        skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)

        return skips_tilde_1, skips_tilde_2, skips_tilde_3


class ResidualEncoderLayer_3(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()

        # combi 2
        self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        
        self.enc_layer_2 = EncoderLayer(d_time, actual_d_feature, 2 * channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)

        # cmobi 2
        self.conv_layer = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)
        self.cond_proj = Conv1d_with_init_saits_new(d_model, channels, 1)

        # before combi 2
        self.conv_cond = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.res_proj = Conv1d_with_init_saits_new(channels, d_model, 1)
        self.skip_proj = Conv1d_with_init_saits_new(channels, d_model, 1)


    # new_design
    def forward(self, x, cond, diffusion_emb):
        # x Noise
        # L -> feature
        # K -> time
        B, K, L = x.shape

        x_proj = torch.transpose(x, 1, 2) # (B, L, K)
        x_proj = self.init_proj(x_proj)

        cond = torch.transpose(cond, 1, 2) # (B, L, K)
        cond = self.cond_proj(cond)
 

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_proj + diff_proj + cond

        y = torch.transpose(y, 1, 2) # (B, K, channels)
        y, _ = self.enc_layer_1(y)
        y = torch.transpose(y, 1, 2)

        y = self.conv_layer(y)
        c_y = self.conv_cond(cond)
        y = y + c_y

        y = torch.transpose(y, 1, 2) # (B, K, 2*channels)
        y, attn_weights_2 = self.enc_layer_2(y)
        y = torch.transpose(y, 1, 2)
 

        y1, y2 = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(y1) * torch.tanh(y2) # (B, channels, K)

        residual = self.res_proj(out) # (B, L, K)
        residual = torch.transpose(residual, 1, 2) # (B, K, L)


        skip = self.skip_proj(out) # (B, L, K)
        skip = torch.transpose(skip, 1, 2) # (B, K, L)

        attn_weights = attn_weights_2 #torch.softmax(attn_weights_1 + attn_weights_2, dim=-1)

        return (x + residual) * math.sqrt(0.5), skip, attn_weights

