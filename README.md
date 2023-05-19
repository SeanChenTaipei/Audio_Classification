# Audio_Classification
AI CUP 2023, Audio Classification

## Environment
```shell
## If you want to run in a virtual env
conda create --name audio python=3.9
conda activate audio
pip install -r requirements.txt
```

## Preprocessing
```
usage: WavEncoder.py [-h] [--train_wav TRAIN_WAV] [--train_csv TRAIN_CSV] [--public_wav PUBLIC_WAV] [--public_csv PUBLIC_CSV] [--private_wav PRIVATE_WAV] [--private_csv PRIVATE_CSV]
                     [--output_path OUTPUT_PATH]

options:
  -h, --help            show this help message and exit
  --train_wav TRAIN_WAV
                        The path of directory of the training audio datas.
  --train_csv TRAIN_CSV
                        The path of the training csv.
  --public_wav PUBLIC_WAV
                        The path ofdirectory of the training audio datas.
  --public_csv PUBLIC_CSV
                        The path of the public csv.
  --private_wav PRIVATE_WAV
                        The path ofdirectory of the training audio datas.
  --private_csv PRIVATE_CSV
                        The path of the private csv.
  --output_path OUTPUT_PATH
                        Path to store the generation.
```
```bash
python WavEncoder.py --train_wav ./Training\ Dataset/training_voice_data \
                     --public_wav ./Public\ Testing\ Dataset/test_data_public \
                     --private_wav ./Private\ Testing\ Dataset/test_data_private \
                     --train_csv ./Training\ Dataset/training\ datalist.csv \
                     --public_csv ./Public\ Testing\ Dataset/test_datalist_public.csv \
                     --private_csv ./Private\ Testing\ Dataset/test_datalist_private.csv \
                     --output_path ./Dataset
```
