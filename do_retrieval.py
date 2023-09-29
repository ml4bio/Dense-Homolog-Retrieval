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

ckpt_path = "./cpu_model/fastmsa-cpu.ckpt"   #-> modified by Sheng Wang at 2022.06.14
input_path = "./input_test.fasta"
qjackhmmer = "./bin/qjackhmmer"
out_path = "./testout/"
search_batch = 10
tar_num = 400000
iter_num = 1

parser = argparse.ArgumentParser(description='fastMSA do homolog retrieval.')
parser.add_argument("-i", "--input_path", default=input_path, help="path of the fasta file containing query sequences")
parser.add_argument("-d", "--database_path", default="./output/agg/", help="path of dir containing database embedding and db converted to DataFrame")
parser.add_argument("-o", "--output_path", default=out_path, help="path to output msas")
parser.add_argument("-n", "--num", default=tar_num, help="retrieve num")
parser.add_argument("-r", "--iters", default=iter_num, help="num of iters by QJackHMMER")


def gen_query(fasta_file_path, out_dir):
    os.makedirs(os.path.join(out_dir, "seq"), exist_ok=True)
    df = ph.read_fasta(fasta_file_path, use_uids=False)
    tot_num = len(df)
    for i in range(tot_num):
        seq_slice = df.iloc[i]
        filename = seq_slice.id
        seq_slice.phylo.to_fasta(os.path.join(out_dir, 'seq', filename+'.fasta'), id_col='id')

def my_align(out_dir, iter_num):
    qlist = os.listdir(os.path.join(out_dir, 'seq'))
    os.makedirs(os.path.join(out_dir, "res"), exist_ok=True)
    for fp in qlist:
        qid = fp.split('.')[0]
        cmd = "%s -B %s --noali --incE 1e-3 -E 1e-3 --cpu 32 -N %d --F1 0.0005 --F2 5e-05 --F3 5e-07 > /dev/null"%(qjackhmmer, str(os.path.join(out_dir, "res", "%s.a3m"%qid)), iter_num, str(os.path.join(out_dir, "seq", "%s.fasta"%qid)), str(os.path.join(out_dir, "res", "%s.fasta"%qid)))

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_path
    out_path = args.output_path 
    tar_num = args.num
    iter_num = args.iters
    idx_path = os.path.join(args.database_path, "index-ebd.index")
    dm_path = os.path.join(args.database_path, "df-ebd.pkl")

    s0 = time.time()

    # print("Start mkdir!!!")
    gen_query(input_path, out_path)
    s1 = time.time()
    # print("Mkdir output cost: %f s"%(s1-s0))

    index = faiss.read_index(idx_path)
    s2 = time.time()
    # print("Load index cost: %f s"%(s2-s1))
    df = pd.read_pickle(dm_path)

    model = MyEncoder.load_from_checkpoint(checkpoint_path=ckpt_path)
    ds = PdDataModule(input_path, 40, model.alphabet)
    
    s3 = time.time()
    # print("Load ckp cost: %f s"%(s3-s2))
    trainer = pl.Trainer() # gpus=[0])
    ret = trainer.predict(model, datamodule=ds, return_predictions=True)
    trainer.save_checkpoint(ckpt_path)
    s4 = time.time()
    # print("Train predict cost: %f s"%(s4-s3))
    # names, qebd = ret[0]
    
    tmp1 = []
    tmp2 = []
    for i in ret:
        n1, q1 = i
        tmp1 += n1
        q1 = torch.tensor(q1).numpy()
        tmp2.append(q1)
    encoded = np.concatenate(tmp2, axis=0)
    # encoded = np.concatenate([t.cpu().numpy() for t in tmp2]) 
    names = tmp1
    # print(encoded.shape)
    
    # encoded = qebd.numpy()
    # print("prepared model")
    s5 = time.time()
    # print("Encode model cost: %f s"%(s5-s4))

    os.makedirs(os.path.join(out_path, "db"), exist_ok=True)
    for i in range(math.ceil(encoded.shape[0]/search_batch)):
        scores, idxes = index.search(encoded[i*search_batch:(i+1)*search_batch], tar_num)
        idx_batch = len(idxes)
        for j in range(idx_batch):
            tar_idx = idxes[j]
            res = df.iloc[tar_idx]
            res.phylo.to_fasta_dev(os.path.join(out_path, "db", names[i*search_batch+j]+'.fasta'))
            
    #end = time.time()
    #print("Time for predict %d : %f s"%(tar_num, end-s5))
    #print("============================================")

    #my_align(out_path, iter_num)

