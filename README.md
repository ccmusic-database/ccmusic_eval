# Evaluation Framework for CCMusic Database Classification Tasks
[![License: MIT](https://img.shields.io/github/license/monetjoe/ccmusic_eval.svg)](https://github.com/monetjoe/ccmusic_eval/blob/main/LICENSE)
[![Python application](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml)
[![hf](https://img.shields.io/badge/HuggingFace-ccmusic-ffd21e.svg)](https://huggingface.co/ccmusic-database)
[![ms](https://img.shields.io/badge/ModelScope-ccmusic-624aff.svg)](https://www.modelscope.cn/organization/ccmusic-database)

Classify spectrograms by fine-tuned pre-trained CNN models.

![](./.github/eval.png)

## Download
```bash
git clone git@github.com:monetjoe/ccmusic_eval.git
cd ccmusic_eval
```

## Environment
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## Usage
```bash
python train.py --ds ccmusic-database/bel_canto --subset eval --data cqt --label singing_method --model squeezenet1_1 --wce True --mode 0
```
### Help
| Args     | Notes                                                                                                            | Options                                                                                                                                                                                    | Type   |
| :------- | :--------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----- |
| --ds     | The dataset on [ModelScope](https://www.modelscope.cn/organization/ccmusic-database?tab=dataset) to be evaluated | For examples: [ccmusic-database/CNPM](https://www.modelscope.cn/datasets/ccmusic-database/CNPM), [ccmusic-database/bel_canto](https://www.modelscope.cn/models/ccmusic-database/bel_canto) | string |
| --subset | The subset of the dataset                                                                                        | For examples: default, eval                                                                                                                                                                | string |
| --data   | Input data colum of the dataset                                                                                  | For examples: mel, cqt, chroma                                                                                                                                                             | string |
| --label  | Label colum of the dataset                                                                                       | For examples: label, singing_method, gender                                                                                                                                                | string |
| --model  | Select a [CV backbone](https://huggingface.co/datasets/monetjoe/cv_backbones) to train                           | [Supported backbones](https://www.modelscope.cn/datasets/monetjoe/cv_backbones/dataPeview)                                                                                                 | string |
| --imgnet | ImageNet version the backbone was pretrained on                                                                  | v1, v2                                                                                                                                                                                     | string |
| --mode   | Training mode ID                                                                                                 | 0=linear_probe, 1=full_finetune, 2=no_pretrain                                                                                                                                             | int    |
| --bsz    | Batch size                                                                                                       | For examples: 1, 2, 4, 8, 16, 32, 64, 128..., default is 4                                                                                                                                 | int    |
| --eps    | Epoch number                                                                                                     | Default is 40                                                                                                                                                                              | int    |
| --wce    | Whether to use weighted cross entropy                                                                            | False, True                                                                                                                                                                                | bool   |

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
