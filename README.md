# Dense Homolog Retriever (DHR)

## Changelog
### 2024-08-22
- Update dependencies in main branch and fix version issue in do_embedding.

## Build Environment

* Clone the repo `git clone https://github.com/heathcliff233/Dense-Homolog-Retrieval.git`
* Go to the directory `cd Dense-Homolog-Retrieval`
* Build using environment.yml   `conda create --name fastMSA --file environment.yml -c pytorch -c conda-forge -c bioconda`
* Activate the environment `conda activate fastMSA`
* Get the customized Phylopandas for fasta processing `git clone https://github.com/heathcliff233/phylopandas.git`


Please download the checkpoints [here](https://drive.google.com/file/d/1t7R_ZQJTIsFM0JVVuY9cLLa9EE2QlIVg/view?usp=sharing) and unzip. We will denote the absolute path to the checkpoint as `$MODEL_PATH`

If you would like a quick test with pre-built index or want to use esm1, please switch to v1 branch.

## Offline Embedding (optional)
* Get the path to sequence database as `$SEQDB_PATH` (require tsv format) and path to output as $OUTPUT_PATH
* Use `python3 do_embedding.py trainer.ur90_path=$SEQDB_PATH model.ckpt_path=$MODEL_PATH hydra.run.dir=$OUTPUT_PATH` to do embedding. Please note that `$SEQDB_PATH` needs to be an absolute path. 
* Aggregate all the result using `python3 do_agg.py -s $SEQDB_PATH -e $OUTPUT_PATH/ebd -o $OUTPUT_PATH/agg`
* For power users, please modify the settings in configuration to allow parallel embedding.

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
If you find it useful, please cite our paper.

```bibtex
@article{Hong2024Aug,
	author = {Hong, Liang and Hu, Zhihang and Sun, Siqi and Tang, Xiangru and Wang, Jiuming and Tan, Qingxiong and Zheng, Liangzhen and Wang, Sheng and Xu, Sheng and King, Irwin and Gerstein, Mark and Li, Yu},
	title = {{Fast, sensitive detection of protein homologs using deep dense retrieval}},
	journal = {Nat. Biotechnol.},
	pages = {1--13},
	year = {2024},
	month = aug,
	issn = {1546-1696},
	publisher = {Nature Publishing Group},
	doi = {10.1038/s41587-024-02353-6}
}
```
