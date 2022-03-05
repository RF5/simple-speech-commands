from typing import Tuple
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import random
import pandas as pd
import librosa
import numpy as np

class UtteranceDS(Dataset):

    def __init__(self, df: pd.DataFrame, sr) -> None:
        super().__init__()
        self.df = df
        self.sr = sr
        self.classes = sorted(self.df['label'].unique().tolist())
        print("Total of", len(self.classes), 'classes')

    def __len__(self): return len(self.df)

    def __getitem__(self, index) -> Tuple[Tensor, str]:
        row = self.df.iloc[index]

        label = self.classes.index(str(row['label']))
        wav, _ = librosa.load(row.path, sr=self.sr, mono=True)
        wav = torch.from_numpy(wav)

        return wav, label

class SpecialCollater():

    def __init__(self, seq_len) -> None:
        self.seq_len = seq_len
        
    def create_batch(self, xs):
        batch_len = self.seq_len
    
        xb = torch.zeros(len(xs), batch_len) 
        lens = torch.zeros(len(xs), dtype=torch.long)
        lbls = torch.empty(len(xs), dtype=torch.long)
        for i in range(len(xs)):
            n_sam = xs[i][0].shape[0]
            if n_sam < batch_len:
                xb[i, :n_sam] = xs[i][0]
                lens[i] = n_sam
            else:
                sp = random.randint(0, n_sam - batch_len)
                xb[i] = xs[i][0][sp:sp+batch_len]
                lens[i] = batch_len
            lbls[i] = xs[i][1]
    
        return xb, lens, lbls

    def __call__(self, xs) -> Tuple[Tensor, Tensor]:
        return self.create_batch(xs)