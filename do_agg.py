import torch 
import os
import pandas as pd
# import phylopandas.phylopandas as ph
from pyarrow import csv
import argparse
import faiss

parser = argparse.ArgumentParser(description='fastMSA aggregate embedding.')
parser.add_argument("-s", "--seqdb_path", default="./input_test.fasta", help="path of the fasta sequence database")
parser.add_argument("-e", "--embdb_path", default="./output/ebd/", help="path of the corresponding embedding output")
parser.add_argument("-o", "--output_path", default="./output/agg/", help="path to output directory for aggregated embeddings")

if __name__ == "__main__":
    args = parser.parse_args()
    seqdb_path = args.seqdb_path
    embdb_path = args.embdb_path
    output_path = args.output_path

    # Load original sequence database
    # seqdb_df = ph.read_fasta(seqdb_path, use_uids=False)
    seqdb_df = csv.read_csv(seqdb_path, 
                        read_options=csv.ReadOptions(column_names=['id', 'sequence']), 
                        parse_options=csv.ParseOptions(delimiter='\t')).to_pandas()
    seqdb_df = seqdb_df.set_index('id')

    # Create Index
    index = faiss.IndexFlatL2(480)
    id_lst = []

    # Load embedded database and process
    for rank in os.listdir(embdb_path):
        for pts in os.listdir(os.path.join(embdb_path, rank)):
            if pts.endswith(".pt"):
                vec = torch.cat(torch.load(os.path.join(embdb_path, rank, pts))[0])
                print(vec.shape)
                index.add(vec.cpu().numpy())

    # Write aggregated results
    os.makedirs(output_path, exist_ok=True)
    seqdb_df.reset_index(inplace=True)
    seqdb_df.to_pickle(os.path.join(output_path, "df-ebd.pkl"))
    faiss.write_index(index, os.path.join(output_path, "index-ebd.index"))
