# DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions



<p align="center">
    <br>
    <img src="https://assets.website-files.com/6392a785ccd80ebf6f060fbe/6392a7df8df82ba809d50347_Logo_Home.svg" width="400"/>
    <br>
<p>




<h4 align="center">
    <p>
        <a href="https://deepbrainai-research.github.io/discohead">Project Page</a> | 
        <a href="https://github.com/deepbrainai-research/koeba">KoEBA Dataset</a> 
    <p>
</h4>


## Requirements

You can install required environments using below commands:

```shell
git clone https://github.com/deepbrainai-research/discohead
cd discohead
conda create -n discohead python=3.7
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
│   ├── grid/
│   │   ├── demo1/
│   │   ├── demo2/
│   ├── koeba/
│   │   ├── demo1/
│   │   ├── demo2/
│   ├── obama/
│   │   ├── demo1/
│   │   ├── demo2/
├── weight/
│   ├── grid.pt
│   ├── koeba.pt
│   ├── obama.pt
├── modules/
‥‥

```
- The `--mode` argument is used to specify which demo video you want to generate:
```shell
python test.py --mode {mode}
```
- Available modes: `obama_demo1, obama_demo2, grid_demo1, grid_demo2, koeba_demo1, koeba_demo2`


## License
```plain
Non-commercial
```
    
   
## Citation 

```plain
@misc{hwang2023discohead,
      title={DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions},
      author={Geumbyeol Hwang and Sunwon Hong and Seunghyun Lee and Sungwoo Park and Gyeongsu Chae},
      year={2023},
      eprint={2303.07697},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}```
 
