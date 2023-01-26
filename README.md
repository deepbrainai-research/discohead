# DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions



<p align="center">
    <br>
    <img src="https://assets.website-files.com/6392a785ccd80ebf6f060fbe/6392a7df8df82ba809d50347_Logo_Home.svg" width="400"/>
    <br>
<p>




<h4 align="center">
    <p>
        <a href="https://deepbrainai-research.github.io/discohead">Demo</a> | 
        <a href="https://github.com/deepbrainai-research/koeba">KoEBA</a> 
    <p>
</h4>


## Requirements
- CUDA 10.2
- PyTorch 1.10.0
- Python 3.7

## Installation
You can install required environments using below commands:

```shell
git clone https://github.com/deepbrainai-research/discohead
cd discohead
conda env create -n discohead python=3.7
conda activate discohead
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Generating Demo Videos

- Download the pre-trained checkpoints from [google drive](https://drive.google.com/drive/folders/1JOWwCVF8v2yNJ_n6a4BsaXuZZFKGo4je?usp=sharing) and put into `weight` folder.
- Download `dataset.zip` from [google drive](https://drive.google.com/drive/folders/1JOWwCVF8v2yNJ_n6a4BsaXuZZFKGo4je?usp=sharing) and unzip into `dataset`. 
- `DisCoHead` directory should have the following structure.

```
DisCoHead/
├── dataset/
│   ├── fig2/
│   ├── fig3/
│   ├── fig4/
├── weight/
│   ├── obama.pt
│   ├── grid.pt
│   ├── koeba.pt
├── modules/
‥‥

```
- The `--fig_number` argument is used for specifying which figure you want to generate.
- To reproduce fig. 2 of obama dataset, run command :

```shell
python test.py --fig_number 2
```

## License
```plain
Non-commercial
```
    
   
## Citation 

```plain
To be updated
```
