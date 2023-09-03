# Audio Classification
AI CUP 2023 - [Multimodal Pathological Voice Classification](https://tbrain.trendmicro.com.tw/Competitions/Details/27) - TEAM_2907

<img width="633" alt="CleanShot 2023-09-03 at 17 30 45@2x" src="https://github.com/jwliao1209/Audio-Classification/assets/55970911/03aae843-789e-47fb-8fc3-87727e73e9ec">

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
## Step 1. Preprocessing
`For reproducing, please jump to Step 2 directly.`
```bash
python WavEncoder.py --train_wav <training_audio_directory> \
                     --public_wav <public_audio_directory> \
                     --private_wav <private_audio_directory> \
                     --train_csv <training_csv_directory> \
                     --public_csv <public_csv_directory> \
                     --private_csv <private_csv_directory> \
                     --output_path <output_path>
```

## Step 2. Reproduce private & public prediction
```bash
bash ./run_reproduce.sh
```
