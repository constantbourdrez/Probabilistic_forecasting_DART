import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import math


def TimeEmbedding(timesteps: th.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = th.exp(th.arange(half_dim, dtype= th.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = th.cat([th.sin(emb), th.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = th.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TemporalAttention(nn.Module):
    """An implementation of the Temporal Attention Module( i.e. compute temporal attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: int, num_of_vertices: int, temporal_length: int):
        super(TemporalAttention, self).__init__()

        self._U1 = nn.Parameter(th.FloatTensor(num_of_vertices))  # for example 307
        self._U2 = nn.Parameter(th.FloatTensor(in_channels, num_of_vertices)) #for example (1, 307)
        self._U3 = nn.Parameter(th.FloatTensor(in_channels))  # for example (1)
        self._be = nn.Parameter(
            th.FloatTensor(1, temporal_length, temporal_length)
        ) # for example (1,12,12)
        self._Ve = nn.Parameter(th.FloatTensor(temporal_length, temporal_length))  #for example (12, 12)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: th.FloatTensor) -> th.FloatTensor:
        """
        Making a forward pass of the temporal attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **E** (PyTorch FloatTensor) - Temporal attention score matrices, with shape (B, T_in, T_in).
        """
        # lhs = left hand side embedding;
        # to calculcate it :
        # permute x:(B, N, F_in, T) -> (B, T, F_in, N)
        # multiply with U1 (B, T, F_in, N)(N) -> (B,T,F_in)
        # multiply with U2 (B,T,F_in)(F_in,N)->(B,T,N)
        # for example (32, 307, 1, 12) -premute-> (32, 12, 1, 307) * (307) -> (32, 12, 1) * (1, 307) -> (32, 12, 307)
        LHS = th.matmul(th.matmul(X.permute(0, 3, 2, 1), self._U1), self._U2) # (32, 12, 307)


        #rhs = right hand side embedding
        # to calculcate it :
        # mutliple U3 with X (F)(B,N,F,T)->(B, N, T)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12)
        RHS = th.matmul(self._U3, X) # (32, 307, 12)

        # Them we multiply LHS with RHS :
        # (B,T,N)(B,N,T)->(B,T,T)
        # for example (32, 12, 307) * (32, 307, 12) -> (32, 12, 12)
        # Then multiply Ve(T,T) with the output
        # (T,T)(B, T, T)->(B,T,T)
        # for example (12, 12) *  (32, 12, 12) ->   (32, 12, 12)
        E = th.matmul(self._Ve, th.sigmoid(th.matmul(LHS, RHS) + self._be))
        E = F.softmax(E, dim=1) #  (B, T, T)  for example (32, 12, 12)
        return E


class SpatialAttention(nn.Module):
    r"""An implementation of the Spatial Attention Module (i.e compute spatial attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(SpatialAttention, self).__init__()

        self._W1 = nn.Parameter(th.FloatTensor(num_of_timesteps))  #for example (12)
        self._W2 = nn.Parameter(th.FloatTensor(in_channels, num_of_timesteps)) #for example (1, 12)
        self._W3 = nn.Parameter(th.FloatTensor(in_channels)) #for example (1)
        self._bs = nn.Parameter(th.FloatTensor(1, num_of_vertices, num_of_vertices)) #for example (1,307, 307)
        self._Vs = nn.Parameter(th.FloatTensor(num_of_vertices, num_of_vertices)) #for example (307, 307)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: th.FloatTensor) -> th.FloatTensor:
        """
        Making a forward pass of the spatial attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **S** (PyTorch FloatTensor) - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        # lhs = left hand side embedding;
        # to calculcate it :
        # multiply with W1 (B, N, F_in, T)(T) -> (B,N,F_in)
        # multiply with W2 (B,N,F_in)(F_in,T)->(B,N,T)
        # for example (32, 307, 1, 12) * (12) -> (32, 307, 1) * (1, 12) -> (32, 307, 12)
        LHS = th.matmul(th.matmul(X, self._W1), self._W2)

        # rhs = right hand side embedding
        # to calculcate it :
        # mutliple W3 with X (F)(B,N,F,T)->(B, N, T)
        # transpose  (B, N, T)  -> (B, T, N)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12) -transpose-> (32, 12, 307)
        RHS = th.matmul(self._W3, X).transpose(-1, -2)

        # Then, we multiply LHS with RHS :
        # (B,N,T)(B,T, N)->(B,N,N)
        # for example (32, 307, 12) * (32, 12, 307) -> (32, 307, 307)
        # Then multiply Vs(N,N) with the output
        # (N,N)(B, N, N)->(B,N,N) (32, 307, 307)
        # for example (307, 307) *  (32, 307, 307) ->   (32, 307, 307)
        S = th.matmul(self._Vs, th.sigmoid(th.matmul(LHS, RHS) + self._bs))
        S = F.softmax(S, dim=1)
        return S # (B,N,N) for example (32, 307, 307)

class Cut(nn.Module):
    def __init__(self, cut_size):
        super().__init__()
        self.cut_size = cut_size
    def forward(self, x):
        return x[:, :, :, : -self.cut_size]


class ConvBlockNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_size=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.cut = Cut(self.padding)
        self.drop =  nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.cut, self.drop)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else None


    def forward(self, X):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(X)
        x_skip = X if self.shortcut is None else self.shortcut(X)

        return out + x_skip


class ResBlock(nn.Module):
    """
        input: (B, in_channels, N, T)
        output:(B, out_channels, N, T)
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,   num_vertices, dropout = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vertices = num_vertices
        self.dropout = dropout
        self.convblock1 = ConvBlockNet(in_channels, out_channels, kernel_size=3, dilation_size=1, dropout= self.dropout)
        self.convblock2 = ConvBlockNet(out_channels, out_channels, kernel_size=3, dilation_size=1, dropout= self.dropout)
        self.t_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
        self.chebconv = ChebConv(out_channels, out_channels, K=2)
        self.t_emb_dim = t_emb_dim
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, (1,1))
        self.norm = nn.LayerNorm(out_channels)


    def forward(self, X, t_emb, edge_index, edge_weight):
        batch_size, num_of_features, num_of_vertices, num_of_timesteps = X.shape # (B, F, N , T)
        out = self.convblock1(X) # (B, out_channels, N, T)
        out += self.t_emb_layers(t_emb)[:,:, None, None]
        out = self.convblock2(out) # (B, out_channels, N, T)
        out = self.norm(out.transpose(1,3)).transpose(1,3) # (B, out_channels, N, T)
        out = out.transpose(1,2)
        X_hat = []
        for t in range(num_of_timesteps):
            X_hat.append(
                th.unsqueeze(self.chebconv(out[:, :, :, t], edge_index, edge_weight = edge_weight),-1)
                )
        X_hat = F.relu(th.cat(X_hat, dim=-1)).transpose(1,2)  # (B, N, out_channels,T) -> (B, out_channels, N, T)
        return X_hat + self.shortcut(X)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channeks, t_emb_dim,   num_vertices, dropout = 0.0):
        """
        :param c_in: in channels, out channels
        :param c_out:
        """
        super().__init__()
        self.res = ResBlock(in_channels, out_channeks, t_emb_dim,   num_vertices, dropout = dropout)

    def forward(self, X, t_emb, edge_index, edge_weight):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        return self.res(X, t_emb, edge_index, edge_weight)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,  kernel_size= (1,3), stride=(1,2), padding=(0,1))

    def forward(self, x: th.Tensor, t: th.Tensor, edge_index, edge_weights):
        _ = t
        _ = edge_index
        _ = edge_weights
        return self.conv(x)


class  UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,   num_vertices, dropout = 0.0):
        super().__init__()
        self.res = ResBlock(in_channels + out_channels, out_channels, t_emb_dim,  num_vertices, dropout = dropout)

    def forward(self, X, t_emb, edge_index, edge_weight):
        return self.res(X, t_emb, edge_index, edge_weight)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, (1, 4), (1, 2), (0, 1))

    def forward(self, x, t, edge_index, edge_weights):
        _ = t
        _ = edge_weights
        _ = edge_index
        return  self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, in_channels, t_emb_dim,   num_vertices, dropout = 0.0):
        super().__init__()
        self.res1 = ResBlock(in_channels, in_channels, t_emb_dim,   num_vertices, dropout = dropout)
        self.res2 = ResBlock(in_channels, in_channels, t_emb_dim,   num_vertices, dropout = dropout)

    def forward(self, X, t_emb, edge_index, edge_weight):
        x = self.res1(X, t_emb, edge_index, edge_weight)

        x = self.res2(X, t_emb, edge_index, edge_weight)

        return x


class UGnet(nn.Module):
    def __init__(self,  in_channels, n_blocks, n_resolutions, t_emb_dim,  num_vertices, temporal_length, proj_dim, channel_multipliers,  dropout = 0.0) -> None:
        super().__init__()
        self.d_h = proj_dim
        self.T = temporal_length
        self.F = in_channels
        self.n_blocks = n_blocks
        self.embedding_dim = t_emb_dim
        self.dropout = dropout


        # first half of U-Net = decreasing resolution
        down = []
        # number of channels
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, t_emb_dim,  num_vertices, dropout = self.dropout))
                in_channels = out_channels

            # down sample at all resolution except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, t_emb_dim,  num_vertices, dropout = self.dropout)

        # #### Second half of U-Net - increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, t_emb_dim,  num_vertices, dropout = self.dropout))

            out_channels = in_channels // channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, t_emb_dim,  num_vertices, dropout = self.dropout))
            in_channels = out_channels
            # up sample at all resolution except last
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.out = nn.Sequential(nn.Conv2d(self.d_h, 1, (1,1)),
                                 nn.Linear(2 * self.T, self.T),)


    def forward(self, X: th.Tensor, t: th.Tensor, c, edge_index, edge_weights):
        """
        :param x: x_t of current diffusion step, (B, F, V, T)
        :param t: diffsusion step
        :param c: condition information
            used information in c:
                x_masked: (B, F, V, T)
        :return:
        """

        X_mask, pos_w, pos_d = c  # x_masked: (B, F, V, T), pos_w: (B,T,1,1), pos_d: (B,T,1,1)


        X = th.cat((X, X_mask), dim=3) # (B, F, V, 2 * T)


        X = self.x_proj(X)

        t = TimeEmbedding(t, self.embedding_dim)

        h = [X]


        for m in self.down:
            X = m(X, t, edge_index, edge_weights)
            h.append(X)

        X = self.middle(X, t, edge_index, edge_weights)

        for m in self.up:
            if isinstance(m,  Upsample):
                X = m(X, t, edge_index, edge_weights)
            else:
                s =h.pop()
                X = th.cat((X, s), dim=1)
                X = m(X,t, edge_index, edge_weights)

        e = self.out(X)
        return e
