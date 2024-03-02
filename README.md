# Multimodal Pathological Voice Classification
This repository is code of AI CUP 2023 Spring [Multimodal Pathological Voice Classification Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/27). We achieved a public ranking of 8th and a private ranking of 1th, corresponding to scores of 0.657057 and 0.641098, respectively.


## Getting the code
You can download all the files in this repository by cloning this repository:
```
git clone https://github.com/jwliao1209/Audio-Classification.git
```


## Proposed pipeline
The feature extraction process consists of two parts. In the first part, we employ the Fast Fourier Transform (FFT) to extract frequency features and calculate statistical indicators, constructing a global feature. The second part involves utilizing a deep learning-based pretraining model to extract local features, followed by dimension reduction using Principal Component Analysis (PCA) to retain relevant feature combinations. For the model training phase, we utilize machine learning-based tree models, namely
Random Forest and LightGBM, and a deep learning-based transformer model called
TabPFN for prediction purposes. An ensemble is performed on the predicted probabilities
these models generate to obtain the final output
<img width="633" alt="CleanShot 2023-09-03 at 17 30 45@2x" src="https://github.com/jwliao1209/Audio-Classification/assets/55970911/03aae843-789e-47fb-8fc3-87727e73e9ec">


## Requirements
To set the environment, you can run this command:
```shell
conda create --name audio python=3.10
conda activate audio
pip install -r requirements.txt
```


## Data Preprocessing
```bash
python WavEncoder.py --train_wav <training_audio_directory> \
                     --public_wav <public_audio_directory> \
                     --private_wav <private_audio_directory> \
                     --train_csv <training_csv_directory> \
                     --public_csv <public_csv_directory> \
                     --private_csv <private_csv_directory> \
                     --output_path <output_path>
```

## Reproducing training results
```bash
bash ./run_reproduce.sh
```

## Operating system and device
We develop the code on Ubuntu 22.04.1 LTS operating system and use python 3.10 version. All trainings are performed on a server with Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz and NVIDIA GeForce GTX 1080 Ti GPU.


## Citation
```
@misc{
    title  = {multimodal_pathological_voice_classification},
    author = {Chun-Hsien Chen, Shu-Cheng Zheng, Jia-Wei Liao, Yi-Cheng Hung},
    url    = {https://github.com/jwliao1209/Audio-Classification},
    year   = {2023}
}
```
