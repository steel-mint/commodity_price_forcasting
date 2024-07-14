import os
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from torch import nn
import lightning as L

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HRC_COLUMN_IDX = -1



class HRCDataset(Dataset):
    def __init__(self, split="train", history_len_weeks=5, future_pred_weeks_len=1, total_buffer=10):
        self.dataset_raw_df = pd.read_csv(
            os.path.join("data", f"{split}.csv")
        )
        self.dataset_raw = self.dataset_raw_df.values.tolist()[total_buffer:]
        self.dataset_raw_df['date'] = pd.to_datetime(self.dataset_raw_df['date'])
        self.dataset_raw_df.set_index('date', inplace=True)
        self.dataset_raw_df.sort_index()
        

        self.history_len_weeks = history_len_weeks
        self.future_pred_weeks_len = future_pred_weeks_len
        self.total_feature_row = history_len_weeks + future_pred_weeks_len
        self.num_features = len(self.dataset_raw[0])

        self.total_buffer = total_buffer
        # batch_idx * history_len_weeks * num_features
        self.num_batchs = int((len(self.dataset_raw) - self.total_feature_row))

        self.feats_normalized = []

        for row in self.dataset_raw:
            start_dt = str(pd.to_datetime(row[0]) - timedelta(days=self.total_buffer*7))
            end_dt = str(pd.to_datetime(row[0]) - timedelta(days=1))
            hist_vals = self.dataset_raw_df.loc[start_dt:end_dt]

            self.feats_normalized.append(self.normalize_feature(
                row, historical_vals=hist_vals, normal_type="mean"
            ))


        self.dataset = torch.empty(
            self.num_batchs,
            self.total_feature_row,
            self.num_features,
            dtype=torch.float32,
        )
        for i in range(self.num_batchs):
            self.dataset[i] = torch.stack(
                [torch.cat([item[0], item[1].view(1)]) for item in self.feats_normalized[i : i + self.total_feature_row]]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        feats_transposed_x = raw_item[:-1, :-1].T

        # hrc_price
        price_y = raw_item[-1][HRC_COLUMN_IDX]
        price_avg = raw_item[:-1, -1].mean()
        return feats_transposed_x, price_y, price_avg

    def normalize_feature(
        self, feature_row: list, historical_vals: pd.DataFrame, normal_type="mean", eps=1e-8
    ):
        if normal_type not in ["first", "mean", "last", "max"]:
            raise ValueError("Invalid Normalization Type")
        
        feature_row.pop(0)
        feature_row = torch.tensor(feature_row)

        if normal_type == "first":
            hist_val = torch.tensor(historical_vals.values.tolist()[0])
        elif normal_type == "mean":
            hist_val = torch.tensor(historical_vals.mean().tolist())
        elif normal_type == "last":
            hist_val = torch.tensor(historical_vals.values.tolist()[-1])
        elif normal_type == "max":
            hist_val = torch.tensor(historical_vals.max().tolist())
        
        return [(feature_row - hist_val) / (hist_val+eps), feature_row[-1]]

class Predictor(L.LightningModule):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.rel_week_pos_encoding = nn.Embedding(5, 1)
        self.feat_transform_linears = nn.ModuleList(
            [nn.Linear(5, 64) for _ in range(11)]
        )
        self.out_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(64, 1), nn.Tanh()) for _ in range(11)]
        )
        self.loss = nn.functional.mse_loss

    def forward(self, x):
        B, F, W = x.shape
        x += self.rel_week_pos_encoding(torch.arange(0, W)).squeeze()
        x = torch.stack([self.feat_transform_linears[i](x[:, i]) for i in range(F)])
        x = self.transformer_encoder(x).reshape(B, F, 64)
        x_out = torch.zeros(len(x),1)
        for i in range(F):
            x_out += self.out_layers[i](x[:, i])
        return x_out

    def training_step(self, batch, batch_idx):
        x, y, avg_cost = batch

        x_out = self.forward(x)
        out = avg_cost.view(len(x), 1) + x_out*avg_cost.view(len(x), 1)
        true = y.view(len(y), 1)
        loss = self.loss(out, true)

        pred_percent = ((out-true)/true)*100
        train_acc = torch.sum(pred_percent.abs() <= 5).item()/len(y)
        
        self.log_dict({"train_loss" : loss, "train_acc" : train_acc}, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, avg_cost = batch

        x_out = self.forward(x)
        out = avg_cost.view(len(x), 1) + x_out*avg_cost.view(len(x), 1)
        true = y.view(len(y), 1)
        loss = self.loss(out, true)
        

        pred_percent = ((out-true)/true)*100
        test_acc = torch.sum(pred_percent.abs() <= 5).item()/len(y)

        self.log_dict({"test_loss" : loss, "test_acc" : test_acc}, on_epoch=True, prog_bar=True, logger=True)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

train_dataset = HRCDataset()
test_dataset = HRCDataset(split='test')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)
wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(accelerator="cpu", max_epochs=10, logger=wandb_logger)
predictor = Predictor()
trainer.fit(model=predictor, train_dataloaders=train_dataloader)
trainer.test(dataloaders=test_dataloader)