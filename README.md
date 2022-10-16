# Dense Homolog Retriever (DHR)

## Build Environment

* build using requirements.txt   `conda create --name fastMSA --file requirements.txt -c pytorch -c conda-forge -c bioconda`
* using /user/liyu/miniconda3/envs/rt directly (if accessible)

note: this repo depends on modified phylopandas


We have one model checkpoint located at `cpu_model/fastmsa-cpu.ckpt` (if there is none, please kindly download [here](https://drive.google.com/file/d/1fRqMwaiWnZ0msW_pp3ircaMIeIVxc4CX/view?usp=sharing))

## Offline Embedding (optional)

* use `python3 pred.py` to do embedding
* aggregate all the result, convert to `numpy.ndarray` and index by `faiss.IndexFlatIP`
* convert raw dataset into DataFrame format according to the sequence order in step 2 and dump into pickle

##  Retrieval

python retrieve.py
usage: retrieve.py [-h] [-i INPUT_PATH] [-d DATABASE_PATH] [-o OUTPUT_PATH] [-n NUM] [-r ITERS]

fastMSA do homolog retrieval.

optional arguments:
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

* input_path: put all query seqs into one fasta file
* output_path: output dir -- seq/db/res, seq subdir contain all queries, db contain retrieved db, res contain all results
* database_path: directory containing database in DataFrame and embedding saved in faiss index. All results produced in Offline Embedding section.


## Running example
/share/wangsheng/miniconda/envs/fastMSA/bin/python retrieve.py -i example/1pazA.fasta -o 1pazA_out -d /share/liyu/hl/fastMSA/ebd/ur90-ebd


## Server Mode Usage

For the server administor, start the server at first
> Note: you don't have to follow this step,  unless the fastMsaApp server(172.16.20.151:7077) is down.
Or Deploy like this:
```
cd /share/linmingzhi/server/fastmsa
nohup /share/wangsheng/miniconda/envs/fastMSA/bin/python manage.py runserver 0.0.0.0:7077 >> nohup_xxx.out 2>&1 &
```

Manage the backend's max workers

Find the max_worker config(FastMsaApp/fm_app.py: line 32), change the value.
The number of max_workers is recommend is `32`.



Send a request by http content: 
```
# Use curl
curl --location --request POST 'http://172.16.20.149:7077/fastmsa' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'input=/user/linmingzhi/project/fastmsa/example/1pazA.fasta' \
--data-urlencode 'output=/user/linmingzhi/output/fastmsa/test026' \
--data-urlencode 'tarnum=320000'
```

## Server Test Mode
```
# Run a server instance hang on
python manage.py runserver 0.0.0.0:7077
```

## One-line Command

For fastMSA only, try the following one-line command:
```
./fastMSA.sh -i example/1pazA.fasta
```

For fastAF2 (fastMSA + standard AF2 with A3M as the input), try below one-line command:
```
./fastAF2.sh -i example/1pazA.fasta
```
