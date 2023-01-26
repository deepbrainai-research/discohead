# DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions

### [Demo Page](https://deepbrainai-research.github.io/discohead)

## The official PyTorch implementation of DisCoHead

## Requirements
- CUDA 10.2
- PyTorch 1.10.0
- Python 3.7

## Installation
- You can install environements by 1) or 2).

* 1) Copy created conda environment

```
git clone  https://github.com/deepbrainai-research/discohead
cd discohead
conda env create -f discohead.yaml
conda activate discohead
```
* 2) Install requirements yourself

```
git clone  https://github.com/deepbrainai-research/discohead
cd discohead
conda env create -n discohead python=3.7
conda activate discohead
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Generate Demo Results

- Download the pre-trained checkpoints.
- Create the ./dataset folder.
- unzip the dataset_demo.zip at ./dataset
- The --mode argument is 