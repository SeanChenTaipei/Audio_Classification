# Audio_Classification
AI CUP 2023, Audio Classification

## Environment
```shell
## If you want to run in a virtual env
conda create --name audio python=3.9
conda activate audio
pip install -r requirements.txt
```
## For reproducing, please go to Step 2 directly.
## Step 1. Preprocessing

```bash
python WavEncoder.py --train_wav ./Training\ Dataset/training_voice_data \
                     --public_wav ./Public\ Testing\ Dataset/test_data_public \
                     --private_wav ./Private\ Testing\ Dataset/test_data_private \
                     --train_csv ./Training\ Dataset/training\ datalist.csv \
                     --public_csv ./Public\ Testing\ Dataset/test_datalist_public.csv \
                     --private_csv ./Private\ Testing\ Dataset/test_datalist_private.csv \
                     --output_path ./Dataset
```
Valid options are
```
-h, --help            
            show this help message and exit
--train_wav
            The path of directory of the training audio datas.
--train_csv
            The path of the training csv.
--public_wav
            The path of directory of the training audio datas.
--public_csv
            The path of the public csv.
--private_wav
            The path of directory of the training audio datas.
--private_csv
            The path of the private csv.
--output_path
            Path to store the generation.
```
## Reproduce private & public prediction
```bash
bash ./run_reproduce.sh
```
