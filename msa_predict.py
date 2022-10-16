import torch
import pytorch_lightning as pl
from mydpr.model.biencoder import MyEncoder
from mydpr.dataset.cath35 import PdDataModule
import sys
import os
import argparse
import faiss
import time
import math
sys.path.append("/share/hongliang") 
import pandas as pd
import numpy as np
import phylopandas.phylopandas as ph
import logging, logging.handlers


logger = logging.getLogger('msaRunner')

dm_path = "/share/liyu/hl/fastMSA/ebd/ur90-ebd/df-ebd.pkl"
idx_path = "/share/liyu/hl/fastMSA/ebd/ur90-ebd/index-ebd.index"
# ckpt_path = "/share/liyu/hl/pl_biencoder-epoch=019-val_acc=0.8651.ckpt"
ckpt_path = "./cpu_model/fastmsa-cpu.ckpt"   #-> modified by Sheng Wang at 2022.06.14
input_path = "/share/liyu/hl/fastMSA/input_test.fasta"
qjackhmmer = "/share/hongliang/qjackhmmer"
out_path = "./testout/"
search_batch = 10
tar_num = 400000
iter_num = 1


class FastMsaApp(object):
    def __init__(self, init_flag=True, p_dm_path=None, p_idx_path=None, p_ckpt_path=None):
        self.index=None
        self.df=None
        self.model=None
        self.idx_path=idx_path
        self.dm_path=dm_path
        self.ckpt_path=ckpt_path
        self.status='Initing'
        self.ti=''
        self.to=''
        if not p_dm_path:
            self.dm_path=dm_path
        if not p_idx_path:
            self.idx_path=idx_path
        if not p_ckpt_path:
            self.ckpt_path=ckpt_path
        if init_flag:
            self.init_msa()
        self.status='Started'
    
    def gen_query(self, fasta_file_path, out_dir):
        oldmask = os.umask(000)
        os.makedirs(os.path.join(out_dir, "seq"), exist_ok=True)
        os.umask(oldmask)
        df = ph.read_fasta(fasta_file_path, use_uids=False)
        tot_num = len(df)
        for i in range(tot_num):
            seq_slice = df.iloc[i]
            filename = seq_slice.id
            seq_slice.phylo.to_fasta(os.path.join(out_dir, 'seq', filename+'.fasta'), id_col='id')

    def my_align(out_dir, iter_num):
        qlist = os.listdir(os.path.join(out_dir, 'seq'))
        oldmask = os.umask(000)
        os.makedirs(os.path.join(out_dir, "res"), exist_ok=True)
        os.umask(oldmask)
        for fp in qlist:
            qid = fp.split('.')[0]
            cmd = "%s -B %s --noali --incE 1e-3 -E 1e-3 --cpu 32 -N %d --F1 0.0005 --F2 5e-05 --F3 5e-07 > /dev/null"%(qjackhmmer, str(os.path.join(out_dir, "res", "%s.a3m"%qid)), iter_num, str(os.path.join(out_dir, "seq", "%s.fasta"%qid)), str(os.path.join(out_dir, "res", "%s.fasta"%qid)))

    def init_msa(self):
        s0 = time.time()
        logger.info("%s Start loading idx and checkpoint..." % time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(s0)))
        self.index = faiss.read_index(self.idx_path)
        self.df = pd.read_pickle(self.dm_path)
        self.model = MyEncoder.load_from_checkpoint(checkpoint_path=self.ckpt_path)
        s1 = time.time()
        logger.info("It cost %f s for loading data -> idx/ckp" % (s1-s0))
    
    def run_predict(self, inputp='', outputp='', tarnump=400000):
        print(' >>> {} {} {} {}submitted'.format(btime(), inputp, outputp, tarnump))
        logger.info("--------->>Start<<--------\n >> input fasta: %s" % inputp)
        s0 = time.time()
        self.gen_query(inputp, outputp)
        ds = PdDataModule(inputp, 40, self.model.alphabet)
        s1 = time.time()
        logger.info("It cost %f s for build pd data module..." % (s1-s0))
        trainer = pl.Trainer()
        ret = trainer.predict(self.model, datamodule=ds, return_predictions=True)
        # trainer.save_checkpoint(ckpt_path)
        tmp1 = []
        tmp2 = []
        for i in ret:
            n1, q1 = i
            tmp1 += n1
            q1 = torch.tensor(q1).cpu().numpy()
            tmp2.append(q1)
        encoded = np.concatenate(tmp2, axis=0)
        names = tmp1
        s2 = time.time()
        print("It cost %f s for trainer.predict()..." % (s2-s1))
        oldmask = os.umask(000)
        os.makedirs(os.path.join(outputp, "db"), exist_ok=True)
        os.umask(oldmask)
        for i in range(math.ceil(encoded.shape[0]/search_batch)):
            s22 = time.time()
            _, idxes = self.index.search(encoded[i*search_batch:(i+1)*search_batch], tarnump)
            s23 = time.time()
            print("It cost %f s for index.search()..." % (s23-s22))
            idx_batch = len(idxes)
            for j in range(idx_batch):
                tar_idx = idxes[j]
                res = self.df.iloc[tar_idx]
                res.phylo.to_fasta_dev(os.path.join(outputp, "db", names[i*search_batch+j]+'.fasta'))
        s3 = time.time()
        print("It cost %f s for search && phylo.to_fasta_dev()...\n<<---------END--------->>" % (s3-s2))
        print('{} {} done'.format(btime(), inputp))
        

def btime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='fastMSA do homolog retrieval.')
    parser.add_argument("-i", "--input_path", default=input_path, help="path of the fasta file containing query sequences")
    parser.add_argument("-d", "--database_path", default="/share/liyu/hl/fastMSA/ebd/ur90-ebd/", help="path of dir containing database embedding and db converted to DataFrame")
    parser.add_argument("-o", "--output_path", default=out_path, help="path to output msas")
    args = parser.parse_args()
    
    input_path = args.input_path
    out_path = args.output_path
    out_path1 = '{}_1'.format(out_path)
    
    idx_path = os.path.join(args.database_path, "index-ebd.index")
    dm_path = os.path.join(args.database_path, "df-ebd.pkl")

    f_msa_app = FastMsaApp(p_dm_path=dm_path, p_idx_path=idx_path)
    f_msa_app.run_predict(inputp=input_path, outputp=out_path)
