import torch as th
from torch import nn
import torch.nn.functional as F
import lightning as L
import torch_geometric_temporal as tgt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from preprocessing import Preprocessor_Graph
from metrics import *


class LightningDART_STG(L.LightningModule):
    def __init__(self, DART_STG, preprocessor, mask_ratio = 0.1):
        super().__init__()
        self.model = DART_STG
        self.preprocessor = preprocessor
        self.loss_fn = self.model.loss
        self.batch_size = 32
        self.mask_ratio = 0.1
        self.horizon_pred = self.model.horizon_pred
        self.data_mean = self.preprocessor.data_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type(th.float32).to(self.model.device)
        self.data_std = self.preprocessor.data_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type(th.float32).to(self.model.device)
        self.mask_ratio = mask_ratio
        self.save_hyperparameters(ignore=['DART_STG'])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        future, history, edge_index, edge_weights = batch.y, batch.x, batch.edge_index[:,:,0], batch.edge_attr[0,:]
        x = th.cat((history, future), dim=3)
        zero = - self.data_mean / self.data_std
        zero = zero[0, -1, 0, 0]
        #print(x)
        x_masked = th.cat((history, future), dim=3)
        x_masked[:, :, -1 , -self.model.horizon_pred:] = zero
        x, x_masked = x.transpose(2,1), x_masked.transpose(2,1)
        loss = 10 * self.loss_fn(x, (x_masked, 0, 0), edge_index, edge_weights)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        future, history, edge_index, edge_weights = batch.y, batch.x, batch.edge_index[:,:,0], batch.edge_attr[0,:]
        x = th.cat((history, future), dim=3)
        zero = - self.data_mean / self.data_std
        zero = zero[0, -1, 0, 0]
        x[:, :, -1 , -self.horizon_pred:] = zero
        x, future = x.transpose(2,1), future.transpose(2,1)
        x_hat = self.model((x, 0, 0), edge_index, edge_weights, n_samples = self.num_samples).to(self.model.device)
        x_hat = x_hat * self.data_std.unsqueeze(0) + self.data_mean.unsqueeze(0)
        future_pred = x_hat[:, :, -1:, :, - self.horizon_pred:]
        future = future * self.data_std + self.data_mean
        future = future[:, -1, :, : ]
        future, future_pred = future.transpose(1,2).cpu(), future_pred.squeeze(2).transpose(2,3).cpu()
        future_pred = th.clamp(future_pred, min=0)
        eval_points = np.ones_like(future)
        test_crps = calc_quantile_CRPS(target = future, forecast=future_pred, eval_points = eval_points)
        test_mae = calc_mae(target = future, forecast=future_pred, eval_points = eval_points)
        self.log("test_crps", test_crps, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_mae", test_mae, on_step=True, on_epoch=True, prog_bar=True)
        return test_crps


    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        future, history, edge_index, edge_weights = batch.y, batch.x, batch.edge_index[:,:,0], batch.edge_attr[0,:]
        x = th.cat((history, future), dim=3)
        zero = - self.data_mean / self.data_std
        zero = zero[0, -1, 0, 0]
        x_masked = th.cat((history, future), dim=3)
        x_masked[:, :, -1 , -self.model.horizon_pred:] = zero
        x, x_masked = x.transpose(2,1), x_masked.transpose(2,1)
        val_loss = 10 * self.loss_fn(x, (x_masked, 0, 0), edge_index, edge_weights)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss


    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=0.002)
        return optimizer

    def show_input(self, batch, batch_idx):
        future, history, edge_index, edge_weights = batch.y, batch.x, batch.edge_index[:,:,0], batch.edge_attr[0,:]
        x = th.cat((history, future), dim=3)
        x_masked = th.cat((history, future), dim=3)
        zero = - self.data_mean / self.data_std
        zero = zero[0, -1, 0, 0]
        x_masked[:, :, -1 , -self.model.horizon_pred:] = zero
        x, x_masked = x.transpose(2,1), x_masked.transpose(2,1)
        return x_masked

    def forward(self, batch, batch_idx, n_samples = 20):
        future, history, edge_index, edge_weights = batch.y, batch.x, batch.edge_index[:,:,0], batch.edge_attr[0,:]
        x = th.cat((history, future), dim=3)
        zero = - self.data_mean / self.data_std
        zero = zero[0, -1, 0, 0]
        x[:, :, -1 , -self.model.horizon_pred:] = zero
        x = x.transpose(2,1)
        x_hat = self.model((x, 0, 0), edge_index, edge_weights, n_samples)
        #x_hat = x_hat * (self.data_max - self.data_min) + self.data_min
        return x_hat
