import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AccountSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features, seq_len=20, target_col='target'):
        self.df = df.copy()
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.features = features
        self.seq_len = seq_len
        self.target_col = target_col
        self.sequences = []
        for acc, grp in self.df.groupby('account'):
            grp = grp.sort_values('time')
            X = grp[features].fillna(0).to_numpy(np.float32)
            y = grp[target_col].astype(int).to_numpy()
            for i in range(len(grp)):
                s = X[max(0, i - seq_len):i]
                if len(s) < seq_len:
                    pad = np.zeros((seq_len-len(s), X.shape[1]), dtype=np.float32)
                    s = np.vstack([pad, s])
                self.sequences.append((s, y[i]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.from_numpy(x), torch.tensor(y)
