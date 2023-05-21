# Audio_Classification
AI CUP 2023, Audio Classification

## Environment
### Hardware env
| Type | Name                                      |
| ---- |:----------------------------------------- |
| os   | Ubuntu 22.04.1 LTS                        |
| cpu  | Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz |
| gpu  | NVIDIA GeForce GTX 1080 Ti                |
### Conda env
If you want to run in a virtual env,
```shell
conda create --name audio python=3.10
conda activate audio
pip install -r requirements.txt
```
---
## Step 1. Preprocessing (For reproducing, please jump to Step 2 directly.)
```bash
python WavEncoder.py --train_wav ./Training\ Dataset/training_voice_data \
                     --public_wav ./Public\ Testing\ Dataset/test_data_public \
                     --private_wav ./Private\ Testing\ Dataset/test_data_private \
                     --train_csv ./Training\ Dataset/training\ datalist.csv \
                     --public_csv ./Public\ Testing\ Dataset/test_datalist_public.csv \
                     --private_csv ./Private\ Testing\ Dataset/test_datalist_private.csv \
                     --output_path ./Dataset
```

## Step 2. Reproduce private & public prediction
```bash
bash ./run_reproduce.sh
```
