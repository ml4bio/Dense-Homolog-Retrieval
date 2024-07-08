# Dense Homolog Retriever (DHR)

## Build Environment

* Clone the repo `git clone https://github.com/heathcliff233/Dense-Homolog-Retrieval.git`
* Go to the directory `cd Dense-Homolog-Retrieval`
* Build using requirements.txt   `conda create --name fastMSA --file requirements.txt -c pytorch -c conda-forge -c bioconda`
* Activate the environment `conda activate fastMSA`
* Get the customized Phylopandas for fasta processing `git clone https://github.com/heathcliff233/phylopandas.git`


We have one model checkpoint located at `cpu_model/fastmsa-cpu.ckpt` (if there is none, please kindly download [here](https://drive.google.com/file/d/1fRqMwaiWnZ0msW_pp3ircaMIeIVxc4CX/view?usp=sharing)). We will denote the absolute path to the checkpoint as `$MODEL_PATH`

## Offline Embedding (optional)
* Get the path to sequence database as `$SEQDB_PATH` (require fasta format) and path to output as $OUTPUT_PATH
* Use `python3 do_embedding.py trainer.ur90_path=$SEQDB_PATH model.ckpt_path=$MODEL_PATH hydra.run.dir=$OUTPUT_PATH` to do embedding. Please note that `$SEQDB_PATH` needs to be an absolute path. 
* Aggregate all the result using `python3 do_agg.py -s $SEQDB_PATH -e $OUTPUT_PATH/ebd -o $OUTPUT_PATH/agg`

##  Retrieval

`python3 do_retrieval.py`
usage: do_retrieval.py [-h] [-i INPUT_PATH] [-d DATABASE_PATH] [-o OUTPUT_PATH] [-n NUM] [-r ITERS]

fastMSA do homolog retrieval.

optional arguments:
```
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        path of the fasta file containing query sequences
  -d DATABASE_PATH, --database_path DATABASE_PATH
                        path of dir containing database embedding and db converted to DataFrame
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to output msas
  -n NUM, --num NUM     retrieve num
  -r ITERS, --iters ITERS
                        num of iters by QJackHMMER
```

* input_path: put all query seqs into one fasta file
* output_path: output dir -- seq/db/res, seq subdir contain all queries, db contain retrieved db, res contain all results
* database_path: directory containing database in DataFrame and embedding saved in faiss index. All results produced in Offline Embedding section.

## Structure prediction (Optional)
Install ColabFold
```
pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold
```
Run batch prediction
```
colabfold_batch $MSA_DIR $PREDICTION_RES
```
## Publication
If you find it useful, please cite us.

Liang Hong, Zhihang Hu, Siqi Sun, Xiangru Tang, Jiuming Wang, Qingxiong Tan, Liangzhen Zheng, Sheng Wang, Sheng Xu, Irwin King, Mark Gerstein, and Yu Li. Fast, sensitive detection of protein homologs using deep dense retrieval. doi: https://doi.org/10.1101/2021.12.20.473431
