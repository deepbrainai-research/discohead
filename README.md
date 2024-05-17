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

- Download the pre-trained checkpoints from [google drive](https://drive.google.com/file/d/1ki8BsZ3Yg2i5OhHF04ULwtgFg6r5Tsro/view?usp=sharing) and put into `weight` folder.
- Download `dataset.zip` from [google drive](https://drive.google.com/file/d/1xy9pxgQYrl2Bnee4npq88zdrHlIcX2wf/view?usp=sharing) and unzip into `dataset`. 
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

<p align=center>
    <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">
        <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/>
    </a>
    <br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. You must not use this work for commercial purposes. You must not distribute it in modified material. You must give appropriate credit and provide a link to the license.
</p>

    
   
## Citation 

```plain
@INPROCEEDINGS{10095670,
  author={Hwang, Geumbyeol and Hong, Sunwon and Lee, Seunghyun and Park, Sungwoo and Chae, Gyeongsu},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095670}}
```
 
