# Multimodal Pathological Voice Classification

This repository contains the code used in the AI CUP 2023 Spring [Multimodal Pathological Voice Classification Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/27), where we achieved 8th place in the public ranking and 1st place in the private ranking, with scores of 0.657057 and 0.641098, respectively.


## Getting the Code
You can download all the files in this repository by cloning it using the following command:
```
git clone https://github.com/jwliao1209/Audio-Classification.git
```


## Proposed Pipeline
The feature extraction process consists of two key steps:
- **Global Feature Extraction:** We utilize Fast Fourier Transform (FFT) to extract frequency features and compute statistical indicators, constructing global features.
- **Local Feature Extraction:** A pre-trained deep learning model is used to extract local features, followed by dimensionality reduction using Principal Component Analysis (PCA) to retain relevant feature combinations.

For the model training phase, we apply machine learning-based tree models such as Random Forest and LightGBM, along with a transformer-based deep learning model called TabPFN. An ensemble method is then used to combine the predicted probabilities from these models, resulting in the final output.

<img width="633" alt="CleanShot 2023-09-03 at 17 30 45@2x" src="https://github.com/jwliao1209/Audio-Classification/assets/55970911/03aae843-789e-47fb-8fc3-87727e73e9ec">


## Requirements
To set up the environment, run the following commands:
```shell
conda create --name audio python=3.10
conda activate audio
pip install -r requirements.txt
```


## Data Preprocessing
To preprocess the dataset, run the command:
```bash
python process_data.py
```


## Training
To start training the models, run the command:
```bash
bash train.py
```


## Inference
For inference, run the following command:
```
python inference.py
```


## Operating system and device
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
```
@misc{chen_multimodal_pathological_voice_classification_2023,
    title  = {Multimodal Pathological Voice Classification},
    author = {Chun-Hsien Chen, Shu-Cheng Zheng, Jia-Wei Liao, Yi-Cheng Hung},
    url    = {https://github.com/jwliao1209/Multimodal-Pathological-Voice-Classification},
    year   = {2023}
}
```
