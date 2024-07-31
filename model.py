import os
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from torch import nn
import lightning as L

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

data_config = {
    "split": "train",
    "history_len_weeks": 5,
    "future_pred_weeks_len": 1,
    "total_buffer": 10,
    "normal_type": "mean",
    "eps": 1e-8,
    "batch_size": 4,
}

predictor_config = {
    "encoder_d": 64,
    "encoder_nheads": 4,
    "transformer_num_layers": 4,
    "context_len_week": 5,
    "num_features": 18,
    "delta_p_wt": 1,
    "ps_wt": 1,
    "as_wt": 0.1,
    "threshold": 1.5,
    "lr": 1e-2,
}


class HRCDataset(Dataset):

    def __init__(self, data_config):
        self.dataset_raw_df = pd.read_csv(
            os.path.join("data", f"{data_config['split']}.csv")
        )
        self.dataset_raw = self.dataset_raw_df.values.tolist()[
            data_config["total_buffer"] :
        ]
        self.dataset_raw_df["date"] = pd.to_datetime(self.dataset_raw_df["date"])
        self.dataset_raw_df.set_index("date", inplace=True)
        self.dataset_raw_df.sort_index()

        self.history_len_weeks = data_config["history_len_weeks"]
        self.future_pred_weeks_len = data_config["future_pred_weeks_len"]
        self.total_feature_row = self.history_len_weeks + self.future_pred_weeks_len
        self.num_features = len(self.dataset_raw[0])

        self.total_buffer = data_config["total_buffer"]
        # batch_idx * history_len_weeks * num_features
        self.num_batchs = int((len(self.dataset_raw) - self.total_feature_row))

        self.feats_normalized = []

        for row in self.dataset_raw:
            start_dt = str(
                pd.to_datetime(row[0]) - timedelta(days=self.total_buffer * 7)
            )
            end_dt = str(pd.to_datetime(row[0]) - timedelta(days=1))
            hist_vals = self.dataset_raw_df.loc[start_dt:end_dt]

            self.feats_normalized.append(
                self.normalize_feature(
                    row,
                    historical_vals=hist_vals,
                    normal_type=data_config["normal_type"],
                    eps=data_config["eps"],
                )
            )

        self.dataset = torch.empty(
            self.num_batchs,
            self.total_feature_row,
            self.num_features,
            dtype=torch.float32,
        )
        for i in range(self.num_batchs):
            self.dataset[i] = torch.stack(
                [
                    torch.cat([item[0], item[1].view(1)])
                    for item in self.feats_normalized[i : i + self.total_feature_row]
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        feats_transposed_x = raw_item[:-1, :-1].T

        y = raw_item[-1][:-1]
        true_price = raw_item[-1][-1]
        last_price = raw_item[-2][-1]
        return feats_transposed_x, y, true_price, last_price

    def normalize_feature(
        self,
        feature_row,
        historical_vals,
        normal_type,
        eps,
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

        return [(feature_row - hist_val) / (hist_val + eps), feature_row[-1]]


class Predictor(L.LightningModule):
    def __init__(self, predictor_config):
        super().__init__()

        self.encoder_d = predictor_config["encoder_d"]
        self.encoder_nheads = predictor_config["encoder_nheads"]
        self.transformer_num_layers = predictor_config["transformer_num_layers"]
        self.context_len_week = predictor_config["context_len_week"]
        self.num_features = predictor_config["num_features"]
        self.delta_p_wt = predictor_config["delta_p_wt"]
        self.ps_wt = predictor_config["ps_wt"]
        self.as_wt = predictor_config["as_wt"]
        self.threshold = predictor_config["threshold"]
        self.lr = predictor_config["lr"]

        self.loss = nn.functional.mse_loss
        self.optimizer = torch.optim.Adam
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_d,
            nhead=self.encoder_nheads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.transformer_num_layers
        )
        self.rel_week_pos_encoding = nn.Embedding(self.context_len_week, 1)
        self.feat_transform_linears = nn.ModuleList(
            [
                nn.Linear(
                    self.context_len_week,
                    self.encoder_d,
                )
                for _ in range(self.num_features)
            ]
        )
        self.out_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.encoder_d, 2), nn.Tanh())
                for _ in range(self.num_features)
            ]
        )

    def forward(self, x):
        B, F, W = x.shape
        x += self.rel_week_pos_encoding(torch.arange(0, W).to(self.device)).squeeze()
        x = torch.stack([self.feat_transform_linears[i](x[:, i]) for i in range(F)])
        x = self.transformer_encoder(x).reshape(B, F, self.encoder_d)
        x_out = torch.stack([self.out_layers[i](x[:, i]) for i in range(F)])
        return x_out

    def training_step(self, batch, batch_idx):
        x, y, true_price, last_price = batch
        x, y, true_price, last_price = (
            x.to(self.device),
            y.to(self.device),
            true_price.to(self.device),
            last_price.to(self.device),
        )

        x_out = self.forward(x)
        x_as = x_out[:, :, 0]
        x_ps = x_out[:, :, 1]
        delta_p = (x_as * x_ps).sum(0)
        true_p = y[:, -1]
        true_ps = y.T
        # Loss 1
        loss_delta_p = self.loss(delta_p, true_p)
        # Loss 2
        loss_ps = self.loss(x_ps, true_ps)
        # Loss 3
        a_sq = (x_as**2).sum(0)
        a_zero = torch.zeros(len(a_sq)).to(self.device)
        loss_as = self.loss(a_sq, a_zero)

        # Final Loss
        loss = (
            self.delta_p_wt * loss_delta_p + self.ps_wt * loss_ps + self.as_wt * loss_as
        )

        out = last_price.view(len(x), 1) + delta_p.view(len(x), 1) * last_price.view(
            len(x), 1
        )
        true = true_price.view(len(true_price), 1)

        pred_percent = ((out - true) / true) * 100
        train_acc = torch.sum(pred_percent.abs() <= self.threshold).item() / len(true)

        self.log_dict(
            {"train_loss": loss, "train_acc": train_acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y, true_price, last_price = batch
        x, y, true_price, last_price = (
            x.to(self.device),
            y.to(self.device),
            true_price.to(self.device),
            last_price.to(self.device),
        )

        x_out = self.forward(x)
        x_as = x_out[:, :, 0]
        x_ps = x_out[:, :, 1]
        delta_p = (x_as * x_ps).sum(0)
        true_p = y[:, -1]
        true_ps = y.T
        # Loss 1
        loss_delta_p = self.loss(delta_p, true_p)
        # Loss 2
        loss_ps = self.loss(x_ps, true_ps)
        # Loss 3
        a_sq = (x_as**2).sum(0)
        a_zero = torch.zeros(len(a_sq)).to(self.device)
        loss_as = self.loss(a_sq, a_zero)

        # Final Loss
        loss = (
            self.delta_p_wt * loss_delta_p + self.ps_wt * loss_ps + self.as_wt * loss_as
        )

        out = last_price.view(len(x), 1) + delta_p.view(len(x), 1) * last_price.view(
            len(x), 1
        )
        true = true_price.view(len(true_price), 1)

        pred_percent = ((out - true) / true) * 100
        test_acc = torch.sum(pred_percent.abs() <= self.threshold).item() / len(true)

        self.log_dict(
            {"test_loss": loss, "test_acc": test_acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    verbose=True,
                    patience=5,
                    factor=0.05,
                ),
                "monitor": "train_acc",
            },
        }


if __name__ == "__main__":
    train_dataset = HRCDataset(data_config)
    test_data_config = data_config.copy()
    test_data_config["split"] = "test"
    test_dataset = HRCDataset(test_data_config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=data_config["batch_size"], shuffle=True, drop_last=True
    )
    wandb_logger = WandbLogger(log_model="all")
    trainer = Trainer(
        accelerator=device,
        max_epochs=1000,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="train_acc", mode="max", patience=50, verbose=True)
        ],
    )
    predictor = Predictor(predictor_config=predictor_config)
    trainer.fit(model=predictor, train_dataloaders=train_dataloader)
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)
