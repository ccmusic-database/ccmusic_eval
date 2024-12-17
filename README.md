# Evaluation Framework for CCMusic Database MSA
[![Python application](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml/badge.svg)](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/monetjoe/ccmusic_eval/blob/msa/LICENSE)

## Download
```bash
git clone -b msa git@github.com:monetjoe/ccmusic_eval.git
cd ccmusic_eval
```

## Requirements
Microsoft Visual C++ 14.0 or greater is required
```bash
conda create -n cv --yes --file conda.txt
conda activate cv
python pip.py
```

## Usage
### Evaluation
Run `eval.py` to evaluate and plot results

### Prerequisites Steps
1. Download and extract [audios](https://www.modelscope.cn/datasets/ccmusic-database/song_structure/resolve/master/data/audio.zip) to `./MSA_dataset/audio`
2. Download and extract [labels](https://www.modelscope.cn/datasets/ccmusic-database/song_structure/resolve/master/data/label.zip) to `./MSA_dataset/Annotations`
3. Run `beat_track.py` first to get beat information, saved to `./MSA_dataset/references`
4. Run `process.py` to perform structure analysis using beat information from `./MSA_dataset/references` to `./MSA_dataset/estimations`
5. Run `txt_to_lab.py` to transform `.txt` to `.lab` as `mir_eval` need `.lab`

## Cite
```bibtex
@dataset{zhaorui_liu_2021_5676893,
  author       = {Monan Zhou, Shenyang Xu, Zhaorui Liu, Zhaowen Wang, Feng Yu, Wei Li and Baoqiang Han},
  title        = {CCMusic: an Open and Diverse Database for Chinese and General Music Information Retrieval Research},
  month        = {mar},
  year         = {2024},
  publisher    = {HuggingFace},
  version      = {1.2},
  url          = {https://huggingface.co/ccmusic-database}
}
```
