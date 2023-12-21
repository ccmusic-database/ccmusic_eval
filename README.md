# Bel-Folk-Classification
[![Python application](https://github.com/monet-joe/ccmusic_clstask_eval/actions/workflows/python-app.yml/badge.svg?branch=genre)](https://github.com/monet-joe/ccmusic_clstask_eval/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/monet-joe/ccmusic_clstask_eval/blob/genre/LICENSE)

Classify singing method by fine-tuned pre-trained CNN models.

## Requirements
```bash
conda create -n cnn --yes --file conda.txt
conda activate cnn
pip install -r requirements.txt
```

## Usage
### Code download
```bash
git clone -b chest-falsetto https://github.com/monet-joe/ccmusic_clstask_eval.git
cd ccmusic_clstask_eval
```

### Train
Assign a backbone(take squeezenet1_1 as an example) after `--model` to start training:
```bash
python train.py --model squeezenet1_1 --fullfinetune True
```
`--fullfinetune True` means full finetune, `False` means linear probing

<a href="https://www.modelscope.cn/datasets/monetjoe/cv_backbones/dataPeview" target="_blank">Supported backbones</a> 

### Plot results
After finishing the training, use below command to plot latest results:
```bash
python plot.py
```

### Predict
Use below command to predict an audio target by latest saved model:
```bash
python eval.py --target ./test/example.wav
```

## Cite
```bash
@dataset{zhaorui_liu_2021_5676893,
  author       = {Zhaorui Liu, Monan Zhou, Shenyang Xu, Zhaowen Wang, Wei Li and Zijin Li},
  title        = {CCMUSIC DATABASE: A Music Data Sharing Platform for Computational Musicology Research},
  month        = {nov},
  year         = {2021},
  publisher    = {Zenodo},
  version      = {1.1},
  doi          = {10.5281/zenodo.5676893},
  url          = {https://doi.org/10.5281/zenodo.5676893}
}
```
