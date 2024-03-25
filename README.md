# Evaluation Framework for CCMusic Database Classification Tasks
[![Python application](https://github.com/monet-joe/ccmusic_eval/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monet-joe/ccmusic_eval/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/monet-joe/ccmusic_eval/blob/main/LICENSE)

Classify spectrograms by fine-tuned pre-trained CNN models.

## Download
```bash
git clone git@github.com:monet-joe/ccmusic_eval.git
cd ccmusic_eval
```

## Requirements
```bash
conda create -n cv --yes --file conda.txt
conda activate cv
pip install -r requirements.txt
```

## Supported backbones
<https://www.modelscope.cn/datasets/monetjoe/cv_backbones/dataPeview>  

## Cite
```bibtex
@dataset{zhaorui_liu_2021_5676893,
  author       = {Monan Zhou, Shenyang Xu, Zhaorui Liu, Zhaowen Wang, Feng Yu, Wei Li and Zijin Li},
  title        = {CCMusic: an Open and Diverse Database for Chinese and General Music Information Retrieval Research},
  month        = {mar},
  year         = {2024},
  publisher    = {HuggingFace},
  version      = {1.2},
  url          = {https://huggingface.co/ccmusic-database}
}
```
