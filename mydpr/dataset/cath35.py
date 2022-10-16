import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from typing import Sequence, Tuple, List, Union
import numpy as np
import pandas as pd
import random
import re
import os
import linecache
from typing import List
import pytorch_lightning as pl
import sys 
sys.path.append("/share/hongliang")
import phylopandas.phylopandas as ph

class SingleConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        limit_size = 400
        batch_size = len(raw_batch)
        max_len = max(len(seq) for id, seq in raw_batch)
        max_len = min(limit_size, max_len)
        ids = []
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, (id, seq_str) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str[:limit_size]], dtype=torch.int64)
            ids.append(id)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str), max_len) + int(self.alphabet.prepend_bos),
            ] = seq1
            if self.alphabet.append_eos:
                tokens[i, min(len(seq_str), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return ids, tokens
        #return tokens


class PdDataset(Dataset):
    def __init__(self, data_path: str):
        self.records = ph.read_fasta(data_path, use_uids=False) 
        #self.records = pd.read_pickle(data_path)
    
    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        rec = self.records.iloc[index]
        return rec['id'], rec['sequence']


class PdDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, alphabet):
        self.path = data_path
        self.batch_size = batch_size
        self.batch_converter = SingleConverter(alphabet)

    def setup(self, stage):
        self.pd_set = PdDataset(self.path)

    def predict_dataloader(self):
        return DataLoader(dataset=self.pd_set, collate_fn=self.batch_converter, num_workers=4, batch_size=self.batch_size) 