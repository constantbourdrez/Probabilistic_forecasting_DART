import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .Unet import UGnet
from typing import Tuple, Optional

class NoiseScheduler:
    def __init__(self, num_time_steps, beta_start, beta_end, beta_schedule, device):
        self.num_time_steps = num_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = device

        if self.beta_schedule ==  'uniform':
            self.betas = th.linspace(self.beta_start, self.beta_end, self.num_time_steps + 1).to(device)

        elif self.beta_schedule == 'quad':
            self.betas = (th.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_time_steps + 1) ** 2).to(device)

        elif self.beta_schedule == 'cosine':
            self.betas = th.linspace(self.beta_start, self.beta_end, self.num_time_steps + 1).to(device)
            self.betas = 0.5 * (1.0 - th.cos(self.betas * np.pi)).to(device)

        else:
            raise NotImplementedError

        self.alphas = (1.0 - self.betas).to(device)
        self.alpha_cumprod = th.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cumprod = th.sqrt(self.alpha_cumprod).to(device)
        self.sqrt_one_minus_alpha_cumprod = th.sqrt(1.0 - self.alpha_cumprod).to(device)

    def add_noise(self, original, noise, t):
        """
        Sample from  q(x_t|x_0) ~ N(x_t; \sqrt\bar\alpha_t * x_0, (1 - \bar\alpha_t)I)
        """
        tensor_shape = original.shape
        feature_size = tensor_shape[2]
        batch_size = tensor_shape[0]
        self.device = original.device

        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size)

        for _ in range(len(tensor_shape) -1):

            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        sqrt_alpha_cumprod = sqrt_alpha_cumprod.repeat(1, original.size(1), original.size(2), original.size(3))
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.repeat(1, original.size(1), original.size(2), original.size(3))
        sqrt_alpha_cumprod[:,:, :feature_size - 1, : ] = 1
        sqrt_one_minus_alpha_cumprod[:,:, :feature_size - 1, : ] = 0

        return sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise

    def sample_prev_time_step(self, xt, noise_pred, t):
        """
        Sample from p(x_{t-1}|x_t, c)
        """
        x0 = (xt - self.sqrt_one_minus_alpha_cumprod[t] * noise_pred) / self.sqrt_alpha_cumprod[t]
        x0 = th.clamp(x0, -1.0, 1.0)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cumprod[t] ))
        mean = mean / th.sqrt(self.alphas[t])

        if t==0:
            return mean, x0
        else:
            variance = (1 -self.alpha_cumprod[t-1]) * (1 - self.alphas[t]) / (1 -self.alpha_cumprod[t])
            sigma = variance ** 0.5
            z = th.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0


class DART_STG(nn.Module):

    """
    Masked Diffusion Model
    """

    def __init__(self, N,sample_steps, sample_strategy, beta_start, beta_end, in_channels=5,
                 n_blocks= 2, n_resolutions=2, t_emb_dim= 10, num_vertices=24, historic_length=60,
                 horizon_pred = 20, proj_dim=32, channel_multipliers= [1,2], beta_schedule = 'uniform', device = 'cpu', dropout = 0.1):
        super().__init__()


        self.N = N #steps in the forward process
        self.sample_steps = sample_steps # steps in the sample process
        self.sample_strategy = sample_strategy # sampe strategy
        self.beta_start = beta_start
        self.horizon_pred = horizon_pred
        self.historic_length = historic_length
        temporal_length = historic_length + horizon_pred
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = device
        self.dropout = dropout
        self.NoiseScheduler = NoiseScheduler(self.N, self.beta_start, self.beta_end, self.beta_schedule, device = device)

        self.eps_model = UGnet(in_channels= in_channels, n_blocks= n_blocks, n_resolutions=n_resolutions, t_emb_dim= t_emb_dim,
                               num_vertices = num_vertices, temporal_length= temporal_length, proj_dim= proj_dim,
                               channel_multipliers = channel_multipliers, dropout = self.dropout).to(device)


        self.q_xt_x0 = self.NoiseScheduler.add_noise

        self.p_sample = self.NoiseScheduler.sample_prev_time_step

    def compute_alpha(self, beta, t):
        beta = th.cat([th.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a


    # simple strategy for DDIM
    def generalized_steps(self, x, seq, model, c, edge_index, edge_weights, **kwargs):
        with th.no_grad():
            n = x.size(0)
            feature_size = x.size(1)
            seq_next = seq[:-1]
            seq = seq[1:]
            x0_preds = []
            xs = [x]
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (th.ones(n) * i).to(x.device)
                t = t.long()
                next_t = (th.ones(n) * j).to(x.device)
                next_t = next_t.long()
                at = self.NoiseScheduler.alpha_cumprod.index_select(0, t+1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, x.size(1), x.size(2), x.size(3))
                at[:, :feature_size-1, :, :] = 1
                at_next = self.NoiseScheduler.alpha_cumprod.index_select(0, next_t+1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, x.size(1), x.size(2), x.size(3))
                at_next[:, :feature_size-1, :, :] = 1
                xt = xs[-1].to(x.device)
                et = model(xt, t, c,  edge_index, edge_weights)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                        kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c1[:, :feature_size -1,:, : ] = 0
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                c2[:, :feature_size -1,:, : ] = 0
                xt_next = at_next.sqrt() * x0_t + c1 * th.randn_like(x) + c2 * et
                xs.append(xt_next.to(self.device))

        return xs, x0_preds



    def p_sample_loop(self, c, edge_index, edge_weights):
        """
        :param c: is the masked input tensor, (B, T, V, D), in the prediction task, T = T_h + T_p
        :return: x: the predicted output tensor, (B, T, V, D)
        """
        x_masked, _, _ = c
        # B, F, V, T = x_masked.shape
        B, _, V, T = x_masked.shape
        with th.no_grad():
            x = th.randn([B, self.eps_model.F, V, T])#generate input noise
            # Remove noise for $T$ steps
            for t in range(self.n, 0, -1):  #in paper, t should start from T, and end at 1
                t = t - 1 # in code, t is index, so t should minus 1
                if t>0:
                    t_tensor = x.new_full((B, ),t, dtype=th.long, device=self.device)
                    noise_pred = self.eps_model(x , t_tensor, c, edge_index, edge_weights)
                    x = self.p_sample(x, noise_pred, t)
        return  x

    def p_sample_loop_ddim(self, c, edge_index, edge_weights):
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape

        N = self.N
        timesteps = self.sample_steps
        # skip_type = "uniform"
        skip_type = self.beta_schedule
        if skip_type == "uniform":
            skip = N // timesteps
            # seq = range(0, N, skip)
            seq = list(range(0, N, skip))
        elif skip_type == "quad":
            skip = N // timesteps
            # seq = range(0, N, skip)
            seq = list(range(0, N, skip))
        elif skip_type == "cosine":
            skip = N // timesteps
            # seq = range(0, N, skip)
            seq = list(range(0, N, skip))
        else:
            raise NotImplementedError

        x = th.randn([B, self.eps_model.F, V, T], device=self.device) #generate input noise
        xs, x0_preds = self.generalized_steps(x, seq, self.eps_model,  c, edge_index, edge_weights, eta=1 + 2e-3)
        return xs, x0_preds

    def set_sample_strategy(self, sample_strategy):
        self.sample_strategy = sample_strategy

    def set_ddim_sample_steps(self, sample_steps):
        self.sample_steps = sample_steps

    def evaluate(self, input, edge_index, edge_weights, n_samples=2):
        x_masked, _, _ = input
        B, F, V, T = x_masked.shape
        if self.sample_strategy == 'ddim_multi':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)#.to(self.config.device)
            xs, x0_preds = self.p_sample_loop_ddim((x_masked, _, _), edge_index, edge_weights)
            x = xs[-1]
            x = x.reshape(B, n_samples, F, V, T)
            return x # (B, n_samples, F, V, T)
        elif self.sample_strategy == 'ddim_one':
            xs, x0_preds = self.p_sample_loop_ddim((x_masked, _, _), edge_index, edge_weights)
            return xs
            x= xs[-n_samples:]
            x = th.stack(x, dim=1)
            return x
        if self.sample_strategy == 'ddpm':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)  # .to(self.config.device)
            x = self.p_sample_loop((x_masked, _, _), edge_index, edge_weights)
            x = x.reshape(B, n_samples, F, V, T)
            return x  # (B, n_samples, F, V, T)
        else:
            raise  NotImplementedError

    def forward(self, input, edge_index, edge_weights, n_samples=2):

        return self.evaluate(input, edge_index, edge_weights, n_samples)

    def loss(self, x0: th.Tensor, c: Tuple, edge_index, edge_weights):
        """
        Loss calculation
        x0: (B, ...)
        c: The condition, c is a tuple of torch tensor, here c = (feature, pos_w, pos_d)
        """
        #
        t = th.randint(0, self.N, (x0.shape[0],), device=x0.device, dtype=th.long)

        eps = th.randn_like(x0)

        xt = self.q_xt_x0(original = x0, noise = eps, t= t)
        eps_theta = self.eps_model(xt, t, c, edge_index, edge_weights)
        eps = eps[:, -1, :, :].unsqueeze(1)
        #x0_pred = self.forward(c, edge_index, edge_weights)
        #future_pred = x0_pred[-1][:, -1, :, -self.horizon_pred:]
        #future = x0[:, -1, :, -self.horizon_pred:]
        #eps = eps[:, -1, :, :].unsqueeze(1)
        return F.mse_loss(eps, eps_theta) #+ F.mse_loss(future, future_pred) / 10
