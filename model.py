import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datetime import timedelta

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

        self.history_len_weeks = history_len_weeks
        self.future_pred_weeks_len = future_pred_weeks_len
        self.total_feature_row = history_len_weeks + future_pred_weeks_len
        # remove the date column
        self.num_features = len(self.dataset_raw[0]) - 1

        self.total_buffer = total_buffer
        # batch_idx * history_len_weeks * num_features
        self.num_batchs = int((len(self.dataset_raw) - total_buffer)/(self.total_feature_row))
        
        self.feats_normalized = []

        for row in self.dataset_raw:
            start_dt = str(pd.to_datetime(row[0]) - timedelta(days=self.total_buffer*7))
            end_dt = str(pd.to_datetime(row[0]) - timedelta(days=1))
            hist_vals = self.dataset_raw_df.loc[start_dt:end_dt]

            self.feats_normalized.append(self.normalize_feature(
                row, historical_vals=hist_vals, normal_type="mean"
            ))
        
        print(len(self.feats_normalized))

        self.dataset = torch.empty(
            self.num_batchs,
            self.total_feature_row,
            self.num_features,
            dtype=torch.float32,
        )
        for i in range(0, self.num_batchs):
            self.dataset[i] = torch.tensor(
                [item for item in self.feats_normalized[i*self.total_feature_row : (i+1)*self.total_feature_row]]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        feats_transposed_x = raw_item[:-1].T
        feats_normalized_x = torch.zeros_like(feats_transposed_x)

        # hrc_price
        price_y = raw_item[-1][HRC_COLUMN_IDX]
        return feats_normalized_x, price_y

    def normalize_feature(
        self, feature_row: list, historical_vals: pd.DataFrame, normal_type="mean"
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
        
        return (feature_row - hist_val) / (hist_val)
