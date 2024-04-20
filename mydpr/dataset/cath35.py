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
import math
import linecache
from typing import List
import pytorch_lightning as pl
import sys 
#import phylopandas.phylopandas as ph
from pyarrow import csv

class SingleConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        limit_size = 800
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

class ArrowDataset(Dataset):
    def __init__(self, data_path: str):
        self.records = csv.read_csv(data_path, 
                        read_options=csv.ReadOptions(column_names=['id', 'sequence']), 
                        parse_options=csv.ParseOptions(delimiter='\t'))
        self.id = self.records[0]
        self.seq = self.records[1]
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.id[index].as_py(), self.seq[index].as_py()

class PdDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, alphabet, trainer):
        super().__init__()
        self.path = data_path
        self.batch_size = batch_size
        self.batch_converter = SingleConverter(alphabet)
        self.rank = trainer.global_rank
        self.world_size = trainer.world_size

    def setup(self, stage):
        #self.pd_set = PdDataset(self.path)
        self.pd_set = ArrowDataset(self.path)

    def predict_dataloader(self):
        sampler = DistributedProxySampler(self.pd_set, self.world_size, self.rank)
        return DataLoader(dataset=self.pd_set, collate_fn=self.batch_converter, sampler=sampler, num_workers=8, batch_size=self.batch_size) 


def get_filename(sel_path: str) -> List[str]:
    nfile = np.genfromtxt(sel_path, dtype='str').T
    path_list = nfile[0]
    names = [str(name)+'.a3m' for name in path_list]
    lines = nfile[1].astype(np.int32).tolist()
    return names, lines


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):        
        super(DistributedProxySampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = self.rank
        self.total_size = len(self.dataset)
        self.num_samples = int(math.ceil(self.total_size * 1.0 / self.num_replicas))

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]

        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, size_limit1=400, size_limit2=800):
        self.alphabet = alphabet
        self.size_limit1 = size_limit1
        self.size_limit2 = size_limit2

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        #max_len = max(max(len(seq1),len(seq2)) for seq1, seq2 in raw_batch)

        ml1 = max(len(seq1) for seq1, seq2 in raw_batch)
        ml2 = max(len(seq2) for seq1, seq2 in raw_batch)
        ml1 = min(self.size_limit1, ml1)
        ml2 = min(self.size_limit2, ml2)
        #ml1 = self.size_limit1
        #ml2 = self.size_limit2
        tokens1 = torch.empty(
            (
                batch_size,
                ml1 + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens2 = torch.empty(
            (
                batch_size,
                ml2 + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens1.fill_(self.alphabet.padding_idx)
        tokens2.fill_(self.alphabet.padding_idx)

        for i, (seq_str1, seq_str2) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens1[i, 0] = self.alphabet.cls_idx
                tokens2[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str1[:ml1]], dtype=torch.int64)
            seq2 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str2[:ml2]], dtype=torch.int64)
            tokens1[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str1), ml1) + int(self.alphabet.prepend_bos),
            ] = seq1
            tokens2[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str2), ml2) + int(self.alphabet.prepend_bos),
            ] = seq2
            if self.alphabet.append_eos:
                tokens1[i, min(len(seq_str1), ml1) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                tokens2[i, min(len(seq_str2), ml2) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return tokens1, tokens2


class  UniclustDataset(Dataset):
    def __init__(self, line_df: pd.DataFrame, data_dir: str):
        self.df = line_df
        self.data_dir = data_dir
        self.sdir = os.listdir(data_dir)
        self.num_sdir = len(self.sdir)

    def get_pair(self, sdir_path: str, fid: int) -> Tuple[str, str]:
        a3m_list = os.listdir(sdir_path)
        # idx1 = random.randint(0, 999)
        idx1 = fid
        fname = a3m_list[idx1]
        fpath = os.path.join(sdir_path, fname)
        fname = fname.split('.')[0]
        tot_lines = self.df.loc[fname].at['lines']//2
        idx2 = random.randint(0, tot_lines-1)
        seq1 = linecache.getline(fpath, 2)
        #seq2 = linecache.getline(fpath, 2)
        seq2 = linecache.getline(fpath, 2*idx2 + 2)

        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        # sdir_path = os.path.join(self.data_dir, self.sdir[index%self.num_sdir])
        sdir_path = os.path.join(self.data_dir, self.sdir[index//1000])
        seq1, seq2 = self.get_pair(sdir_path, index%1000)
        return seq1, seq2

    def __len__(self):
        return 1000*self.num_sdir


class UniclustDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg_dir, batch_size, alphabet):
        super().__init__()
        self.data_dir = data_dir
        self.cfg_dir = cfg_dir
        self.batch_size = batch_size
        self.batch_converter = BatchConverter(alphabet)

    def prepare_data(self):
        pass

    def setup(self, stage):
        train_path = os.path.join(self.cfg_dir, 'train/')
        tr_df = pd.read_table(os.path.join(self.cfg_dir, 'train.txt'), sep=',', index_col=0)
        self.tr_set = UniclustDataset(tr_df, train_path)

        val_path = os.path.join(self.cfg_dir, 'eval/')
        va_df = pd.read_table(os.path.join(self.cfg_dir, 'eval.txt'), sep=',', index_col=0)
        # va_df[va_df['lines'] > 4] = 4
        self.ev_set = UniclustDataset(va_df, val_path)

        test_path = os.path.join(self.cfg_dir, 'test/')
        ts_df = pd.read_table(os.path.join(self.cfg_dir, 'test.txt'), sep=',', index_col=0)
        self.ts_set = UniclustDataset(ts_df, test_path)

        
    def train_dataloader(self):
        return DataLoader(dataset=self.tr_set, collate_fn=self.batch_converter, num_workers=8, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(dataset=self.ev_set, collate_fn=self.batch_converter, num_workers=8, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(dataset=self.ts_set, collate_fn=self.batch_converter, num_workers=8, batch_size=self.batch_size)