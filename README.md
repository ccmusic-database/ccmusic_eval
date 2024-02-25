# Evaluation Framework for CCMusic Database Classification Tasks
[![Python application](https://github.com/monet-joe/ccmusic_clstask_eval/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monet-joe/ccmusic_clstask_eval/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/monet-joe/ccmusic_clstask_eval/blob/main/LICENSE)

Classify spectrograms by fine-tuned pre-trained CNN models.

## Maintenance
```bash
git clone git@gitee.com:MuGeminorum/ccmusic_clstask_eval.git
cd ccmusic_clstask_eval
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
```bash
@dataset{zhaorui_liu_2021_5676893,
  author       = {Monan Zhou, Shenyang Xu, Zhaorui Liu, Zhaowen Wang, Feng Yu, Wei Li and Zijin Li},
  title        = {CCMusic: General Database for Computational Musicology and Chinese Music Technology Research},
  month        = {nov},
  year         = {2021},
  publisher    = {Zenodo},
  version      = {1.1},
  doi          = {10.5281/zenodo.5676893},
  url          = {https://doi.org/10.5281/zenodo.5676893}
}
```
