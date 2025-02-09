# Evaluation Framework for CCMusic Database Detection Task
[![License: MIT](https://img.shields.io/github/license/monetjoe/ccmusic_eval.svg)](https://github.com/monetjoe/ccmusic_eval/blob/main/LICENSE)
[![Python application](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml)
[![hf](https://img.shields.io/badge/HuggingFace-ccmusic-ffd21e.svg)](https://huggingface.co/ccmusic-database)
[![ms](https://img.shields.io/badge/ModelScope-ccmusic-624aff.svg)](https://www.modelscope.cn/organization/ccmusic-database)

Detect spectrograms by fine-tuned pre-trained CNN models.

![](./.github/eval.jpg)

## Download
```bash
git clone -b tech99 git@github.com:monetjoe/ccmusic_eval.git
cd ccmusic_eval
```

## Environment
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

### Fixed Hyper Params
|     Param      | Value |   Range   |
| :------------: | :---: | :-------: |
|   iteration    |  10   |   train   |
|       lr       | 0.001 | optimizer |
|    momentum    |  0.9  | optimizer |
|   optimizer    |  SGD  | scheduler |
|      mode      |  min  | scheduler |
|     factor     |  0.1  | scheduler |
|    patience    |   5   | scheduler |
|    verbose     | True  | scheduler |
|   threshold    |  lr   | scheduler |
| threshold_mode |  rel  | scheduler |
|    cooldown    |   0   | scheduler |
|     min_lr     |   0   | scheduler |
|      eps       | 1e-08 | scheduler |

## Cite
```bibtex
@dataset{zhaorui_liu_2021_5676893,
  author       = {Monan Zhou, Shenyang Xu, Zhaorui Liu, Zhaowen Wang, Feng Yu, Wei Li and Baoqiang Han},
  title        = {CCMusic: an Open and Diverse Database for Chinese Music Information Retrieval Research},
  month        = {mar},
  year         = {2024},
  publisher    = {HuggingFace},
  version      = {1.2},
  url          = {https://huggingface.co/ccmusic-database}
}
```
