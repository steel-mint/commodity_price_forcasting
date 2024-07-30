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

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
HRC_COLUMN_IDX = -1

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
    "upscale_deminsion": 64,
    "loss_fn": nn.functional.mse_loss,
    "threshold": 1.5,
    "optimizer": torch.optim.Adam,
    "lr": 1e-3,
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

        y = raw_item[-1][-2]
        true_price = raw_item[-1][HRC_COLUMN_IDX]
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
        self.upscale_deminsion = predictor_config["upscale_deminsion"]
        self.loss = predictor_config["loss_fn"]
        self.threshold = predictor_config["threshold"]
        self.optimizer = predictor_config["optimizer"]
        self.lr = predictor_config["lr"]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_d, nhead=self.encoder_nheads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.transformer_num_layers
        )
        self.rel_week_pos_encoding = nn.Embedding(self.context_len_week, 1)
        self.feat_transform_linears = nn.ModuleList(
            [
                nn.Linear(self.context_len_week, self.upscale_deminsion)
                for _ in range(self.num_features)
            ]
        )
        self.out_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.upscale_deminsion, 1), nn.Tanh())
                for _ in range(self.num_features)
            ]
        )

    def forward(self, x):
        B, F, W = x.shape
        x += self.rel_week_pos_encoding(torch.arange(0, W).to(self.device)).squeeze()
        x = torch.stack([self.feat_transform_linears[i](x[:, i]) for i in range(F)])
        x = self.transformer_encoder(x).reshape(B, F, self.upscale_deminsion)
        x_out = torch.zeros(len(x), 1).to(self.device)
        for i in range(F):
            x_out += self.out_layers[i](x[:, i])
        return x_out

    def training_step(self, batch, batch_idx):
        x, y, true_price, avg_price = batch
        x, y, true_price, avg_price = (
            x.to(self.device),
            y.to(self.device),
            true_price.to(self.device),
            avg_price.to(self.device),
        )

        x_out = self.forward(x)
        y = y.view(len(y), 1)
        loss = self.loss(x_out, y)

        out = avg_price.view(len(x), 1) + x_out * avg_price.view(len(x), 1)
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
        x, y, true_price, avg_price = batch
        x, y, true_price, avg_price = (
            x.to(self.device),
            y.to(self.device),
            true_price.to(self.device),
            avg_price.to(self.device),
        )

        x_out = self.forward(x)
        y = y.view(len(y), 1)
        loss = self.loss(x_out, y)

        out = avg_price.view(len(x), 1) + x_out * avg_price.view(len(x), 1)
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
        return self.optimizer(self.parameters(), lr=self.lr)


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
    trainer = Trainer(accelerator=device, max_epochs=10, logger=wandb_logger)
    predictor = Predictor(predictor_config=predictor_config)
    trainer.fit(model=predictor, train_dataloaders=train_dataloader)
    trainer.test(dataloaders=test_dataloader)
