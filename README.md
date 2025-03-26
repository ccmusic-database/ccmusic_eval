# MSA Evaluation for CCMusic Database
[![License: MIT](https://img.shields.io/github/license/monetjoe/ccmusic_eval.svg)](https://github.com/monetjoe/ccmusic_eval/blob/main/LICENSE)
[![Python application](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml/badge.svg)](https://github.com/monetjoe/ccmusic_eval/actions/workflows/python-app.yml)
[![hf](https://img.shields.io/badge/HuggingFace-song__structure-ffd21e.svg)](https://huggingface.co/ccmusic-database/song_structure)
[![ms](https://img.shields.io/badge/ModelScope-song__structure-816DF8.svg)](https://www.modelscope.cn/models/ccmusic-database/song_structure)

![image](https://github.com/user-attachments/assets/6f15e10c-6b18-4ed3-b7b4-493b2c260058)

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
@dataset{zhaorui_liu_2021_5654924,
  author    = {Zhaorui Liu and Zijin Li},
  title     = {Music Data Sharing Platform for Academic Research (CCMusic)},
  month     = nov,
  year      = 2021,
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.5654924},
  url       = {https://doi.org/10.5281/zenodo.5654924}
}
```
