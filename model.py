import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HRC_COLUMN_IDX = -1


class HRCDataset(Dataset):
    def __init__(self, split="train", history_len_weeks=5, future_pred_weeks_len=1):
        self.dataset_raw = pd.read_csv(
            os.path.join(BASE_PATH, "data", f"{split}.csv")
        ).values.tolist()
        self.history_len_weeks = history_len_weeks
        self.future_pred_weeks_len = future_pred_weeks_len

        # remove the date column
        self.num_features = len(self.dataset_raw[0]) - 1

        total_buffer = history_len_weeks + future_pred_weeks_len
        # batch_idx * history_len_weeks * num_features
        self.dataset = torch.empty(
            len(self.dataset_raw) - history_len_weeks,
            total_buffer,
            self.num_features,
            dtype=torch.float32,
        )
        for i in range(0, len(self.dataset_raw) - total_buffer):
            self.dataset[i] = torch.tensor(
                [item[1:] for item in self.dataset_raw[0 : 0 + total_buffer]]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        feats_transposed_x = raw_item[:-1].T
        feats_normalized_x = torch.zeros_like(feats_transposed_x)
        for i in range(len(feats_transposed_x)):
            feats_normalized_x[i] = self.normalize_feature(
                feats_transposed_x[i], normal_type="first"
            )

        # hrc_price
        price_y = raw_item[-1][HRC_COLUMN_IDX]
        return feats_normalized_x, price_y

    def normalize_feature(
        self, feature_row: torch.tensor, normal_type="first", eps=1e-8
    ):
        if normal_type not in ["first", "mean", "last", "max"]:
            raise ValueError("Invalid Normalization Type")

        if normal_type == "first":
            scalar_val = feature_row[0]
        elif normal_type == "mean":
            scalar_val = np.mean(feature_row)
        elif normal_type == "last":
            scalar_val = feature_row[-1]
        elif normal_type == "max":
            scalar_val = np.max(feature_row)

        return (feature_row - scalar_val) / (scalar_val + eps)
