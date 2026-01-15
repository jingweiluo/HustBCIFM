<div align="center">

# HustBCIFM


_A Criss-Cross Brain Foundation Model for EEG Decoding_

</div>


<div align="center">
<img src="figure/CBraMod_logo.png" style="width: 15%;" />
</div>


<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚢&nbsp;<a href="#-pretrain">Pretrain</a>
    | ⛵&nbsp;<a href="#-finetune">Finetune</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>

## 🔍 About
A novel EEG foundation model, for EEG decoding on various clinical and BCI application.
<div align="center">

</div>



## 🔨 Setup
example:
```commandline
conda create -n HUSTBCI python=3.10
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 🚢 Pretrain
You can pretrain the model on our pretraining dataset or your custom pretraining dataset using the following code:
```commandline
CUDA_VISIBLE_DEVICES=4 python pretrain_main.py --datasets_dir /home/wengjiayi/data/datasets/BigDownstream/mental-arithmetic/processed --cuda 0 --batch_size 32
```
A pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).Download and put it in pretrained_weights/.

## ⛵ Finetune
You can finetune CBraMod on our selected downstream datasets using the following code:
```commandline
CUDA_VISIBLE_DEVICES=4 python finetune_main.py --datasets_dir /home/wengjiayi/data/datasets/BigDownstream/mental-arithmetic/processed --foundation_dir /home/wengjiayi/BCIFM-model/github_copy/test/HustBCIFM/pretrained_weights/pretrained_weights.pth --cuda 0 --batch_size 32 --seed 1234
```


## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:
```python
import torch
import torch.nn as nn
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
model.proj_out = nn.Identity()
classifier = nn.Sequential(
  Rearrange('b c s p -> b (c s p)'),
  nn.Linear(22*4*200, 4*200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(4 * 200, 200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(200, 4),
).to(device)

# mock_eeg.shape = (batch_size, num_of_channels, time_segments, points_per_patch)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)

# logits.shape = (batch_size, num_of_classes)
logits = classifier(model(mock_eeg))
```



## 🔗 Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
