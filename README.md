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
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## 🚢 Pretrain
You can pretrain the model on our pretraining dataset or your custom pretraining dataset using the following code:
```commandline
python pretrain_main.py
```
A pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).Download and put it in pretrained_weights/.

## ⛵ Finetune
You can finetune CBraMod on our selected downstream datasets using the following code:
```commandline
python finetune_main.py
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
