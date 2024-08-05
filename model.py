import os
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from torch import nn
import lightning as L

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

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
    "train_batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "split_size": 0.2,
}

predictor_config = {
    "encoder_d": 64,
    "encoder_nheads": 4,
    "transformer_num_layers": 4,
    "transformer_dim_feedfoward": 128,
    "context_len_week": 5,
    "num_features": 18,
    "delta_p_wt": 5,
    "ps_wt": 4,
    "as_wt": 1,
    "threshold": 1.0,
    "lr": 1e-2,
}


class HRCDataset(Dataset):

    def __init__(self, data_config):

        self.total_buffer = data_config["total_buffer"]
        self.split = data_config["split"]
        if self.split == "predict":
            self.dataset_raw_df = pd.concat(
                [
                    pd.read_csv("data/train.csv"),
                    pd.read_csv("data/test.csv"),
                    pd.read_csv("data/predict.csv"),
                ]
            )

            self.dataset_raw_df["date"] = pd.to_datetime(self.dataset_raw_df["date"])
            self.dataset_raw_df.set_index("date", inplace=True)
            self.dataset_raw_df.sort_index()

            self.dataset_raw = self.dataset_raw_df[
                self.dataset_raw_df.index
                < (pd.to_datetime(data_config["pred_date"]) - timedelta(days=1))
            ].tail(15)
            self.dataset_raw = self.dataset_raw.reset_index().values.tolist()[
                self.total_buffer :
            ]

        else:
            self.dataset_raw_df = pd.read_csv(os.path.join("data", f"{self.split}.csv"))
            self.dataset_raw = self.dataset_raw_df.values.tolist()[self.total_buffer :]
            self.dataset_raw_df["date"] = pd.to_datetime(self.dataset_raw_df["date"])
            self.dataset_raw_df.set_index("date", inplace=True)
            self.dataset_raw_df.sort_index()

        # batch_idx * history_len_weeks * num_features

        self.feats_normalized = []
        self.num_features = len(self.dataset_raw[0])

        for row in self.dataset_raw:
            start_dt = str(
                pd.to_datetime(row[0]) - timedelta(days=self.total_buffer * 7)
            )
            if self.split == "predict":
                end_dt = str(pd.to_datetime(row[0]) + timedelta(days=1))
            else:
                end_dt = str(pd.to_datetime(row[0]) - timedelta(days=1))

            hist_vals = self.dataset_raw_df.loc[start_dt:end_dt]

            self.feats_normalized.append(
                self.normalize_feature(
                    feature_row=row,
                    historical_vals=hist_vals,
                    normal_type=data_config["normal_type"],
                    eps=data_config["eps"],
                )
            )
        if self.split == "predict":
            self.dataset = torch.empty(
                1,
                data_config["history_len_weeks"],
                self.num_features,
                dtype=torch.float32,
            )
            self.dataset[0] = torch.stack(
                [
                    torch.cat([item[0], item[1].view(1)])
                    for item in self.feats_normalized[
                        0 : data_config["history_len_weeks"]
                    ]
                ]
            )
            return

        self.history_len_weeks = data_config["history_len_weeks"]
        self.future_pred_weeks_len = data_config["future_pred_weeks_len"]
        self.total_feature_row = self.history_len_weeks + self.future_pred_weeks_len
        self.num_batchs = int((len(self.dataset_raw) - self.total_feature_row))

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
        """
        Assume 5 weeks as short history and num features as 18
        Args:
            idx (int): Index of the dataset

        Returns:
            feats_transposed_x (torch.Tensor): 18 features * 5 weeks (18 x 5)
            y (torch.Tensor): Delta feature values of the 6th week (18)
            true_price (torch.Tensor): HRC price of 6th week (1)
            last_price (torch.Tensor): HRC price of 5th week (1)
        """
        raw_item = self.dataset[idx]
        if self.split == "predict":
            feats_transposed_x = raw_item[:, :-1].T
            last_price = raw_item[-1][-1]
            true_price = 0
            y = 0
        else:
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


class DataModule(L.LightningDataModule):

    def __init__(self, data_config, pred_date=None):
        super().__init__()
        self.dataset = HRCDataset(data_config)
        self.data_config = data_config
        self.pred_date = pred_date

    def setup(self, stage: str):
        if stage == "fit":
            self.train, self.val = train_test_split(
                self.dataset, test_size=self.data_config["split_size"], shuffle=True
            )
        if stage == "test":
            test_data_config = self.data_config.copy()
            test_data_config["split"] = "test"
            self.test_dataset = HRCDataset(test_data_config)
        if stage == "predict":
            predict_data_config = self.data_config.copy()
            predict_data_config["split"] = "predict"
            predict_data_config["pred_date"] = self.pred_date
            self.predict_dataset = HRCDataset(predict_data_config)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.data_config["train_batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.data_config["val_batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config["test_batch_size"],
            shuffle=True,
        )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1)


class Predictor(L.LightningModule):
    def __init__(self, predictor_config):
        super().__init__()

        self.encoder_d = predictor_config["encoder_d"]
        self.encoder_nheads = predictor_config["encoder_nheads"]
        self.transformer_num_layers = predictor_config["transformer_num_layers"]
        self.transformer_dim_feedfoward = predictor_config["transformer_dim_feedfoward"]
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
            dim_feedforward=self.transformer_dim_feedfoward,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_num_layers,
            norm=nn.LayerNorm(self.encoder_d),
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
        # Predict two outputs per feature (i. del change and ii. change co-efficient to HRC price)
        self.out_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.encoder_d, 2), nn.Tanh())
                for _ in range(self.num_features)
            ]
        )
        self.save_hyperparameters()

    def forward(self, x):
        # (4 * 18 * 5)
        B, F, W = x.shape
        x += self.rel_week_pos_encoding(torch.arange(0, W).to(self.device)).squeeze()
        # (4 * 18 * 5) -> (18 * 4 * 64)
        x = torch.stack([self.feat_transform_linears[i](x[:, i]) for i in range(F)])
        x = self.transformer_encoder(x).reshape(B, F, self.encoder_d)
        # (4 * 18 * 64) -> (18 * 4 * 2)
        logits = torch.stack([self.out_layers[i](x[:, i]) for i in range(F)])
        return logits

    def loss_n_acc_func(self, logits, y, true_price, last_price):
        """Loss & Accuracy Calculation Function

        Args:
            logits (torch.tensor): Result from self.forward(); shape: (18, 4, 2)
            y (torch.tensor): Real delta feature values of the 6th week; shape: (4 * 18)
            true_price (torch.tensor): Shape: (4)
            last_price (torch.tensor): Shape: (4)

        Returns:
            _type_: _description_
        """
        F, B, _ = logits.shape
        # x_as: delta change in the feature (a1, a2, a3, ..., a18)
        # shape: (18, 4, 1)
        x_as = logits[:, :, 0]
        # x_ps: change co-efficient to HRC price (del p1, del p2, ..., del p18)
        # shape: (18, 4, 1)
        x_ps = logits[:, :, 1]
        # shape: 18 * 4 * 1 -> 4
        delta_p = (x_as * x_ps).sum(0)
        # true_p = change in hrc price (4 * 1)
        true_p = y[:, -1]
        # shape: 18 * 4
        true_ps = y.T
        # Loss 1: Loss for HRC True Price - HRC Predicted Price
        loss_delta_p = self.loss(delta_p, true_p)
        # Loss 2: Loss for each factor for predicted vs original delta
        loss_ps = self.loss(x_ps, true_ps)
        # Loss 3: Energy loss (deviation of coefficients from zero)
        loss_as = (x_as**2).sum(0).mean()

        # Final Loss
        loss = (
            self.delta_p_wt * loss_delta_p + self.ps_wt * loss_ps + self.as_wt * loss_as
        )
        out = last_price.view(B, 1) + delta_p.view(B, 1) * last_price.view(B, 1)
        true = true_price.view(B, 1)

        pred_percent = ((out - true) / true) * 100
        accuracy = torch.sum(pred_percent.abs() <= self.threshold).item() / B

        return loss, loss_delta_p, loss_ps, loss_as, accuracy

    def training_step(self, batch, batch_idx):
        x, y, true_price, last_price = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
        )

        logits = self.forward(x)

        loss, loss_delta_p, loss_ps, loss_as, train_acc = self.loss_n_acc_func(
            logits=logits, y=y, true_price=true_price, last_price=last_price
        )

        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_delta_p": loss_delta_p,
                "train/loss_coeff": loss_as,
                "train/loss_energy": loss_ps,
                "train/acc_total": train_acc,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, true_price, last_price = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
        )

        logits = self.forward(x)

        loss, loss_delta_p, loss_ps, loss_as, val_acc = self.loss_n_acc_func(
            logits=logits, y=y, true_price=true_price, last_price=last_price
        )
        self.log_dict(
            {
                "val/loss_total": loss,
                "val/loss_delta_p": loss_delta_p,
                "val/loss_coeff": loss_as,
                "val/loss_energy": loss_ps,
                "val/acc_total": val_acc,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y, true_price, last_price = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
        )
        logits = self.forward(x)

        loss, loss_delta_p, loss_ps, loss_as, test_acc = self.loss_n_acc_func(
            logits=logits, y=y, true_price=true_price, last_price=last_price
        )
        self.log_dict(
            {
                "test/loss_total": loss,
                "test/loss_delta_p": loss_delta_p,
                "test/loss_coeff": loss_as,
                "test/loss_energy": loss_ps,
                "test/acc_total": test_acc,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, true_price, last_price = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
        )
        print(x.shape)

        logits = self.forward(x)
        F, B, _ = logits.shape
        x_as = logits[:, :, 0]
        x_ps = logits[:, :, 1]
        delta_p = (x_as * x_ps).sum(0)
        pred_out = last_price.view(B, 1) + delta_p.view(B, 1) * last_price.view(B, 1)
        return pred_out

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    verbose=True,
                    patience=5,
                    factor=0.05,
                ),
                "monitor": "val/loss_total",
            },
        }
        return optimizer


if __name__ == "__main__":
    datamodule = DataModule(data_config=data_config)
    wandb_logger = WandbLogger(log_model="all")
    trainer = Trainer(
        accelerator=device,
        devices=2,
        # strategy="ddp",
        # max_epochs=1000,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(
                monitor="val/loss_total", mode="min", patience=50, verbose=True
            )
        ],
    )
    predictor = Predictor(predictor_config=predictor_config)
    trainer.fit(model=predictor, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
