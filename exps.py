import warnings
from train import train

SPECTS = ["mel", "cqt", "chroma"]
SUBSETS = ["eval", "eval2"]  # 8:2 vs 6:4


def erhu_exp(
    backbones=[
        "swin_t",
        "swin_s",
        "convnext_tiny",
        "alexnet",
        "googlenet",
        "squeezenet1_1",
        "densenet201",
    ]
):
    for bb in backbones:
        for spect in SPECTS:
            for subset in SUBSETS:
                train(
                    dataset="ccmusic-database/erhu_playing_tech",
                    subset=subset,
                    data_col=spect,
                    label_col="label",
                    backbone=bb,
                    focal_loss=True,
                    full_finetune=True,
                    epoch_num=40,
                )


def bel_exp(
    backbones=[
        "swin_t",
        "swin_s",
        "convnext_tiny",
        "alexnet",
        "googlenet",
        "squeezenet1_1",
        "mnasnet1_3",
    ]
):
    for bb in backbones:
        for spect in SPECTS:
            for subset in SUBSETS:
                train(
                    dataset="ccmusic-database/bel_canto",
                    subset=subset,
                    data_col=spect,
                    label_col="label",
                    backbone=bb,
                    focal_loss=True,
                    full_finetune=True,
                    epoch_num=40,
                )


def chest_exp(
    backbones=[
        "swin_t",
        "swin_s",
        "alexnet",
        "mnasnet1_3",
        "googlenet",
        "squeezenet1_1",
        "densenet201",
    ]
):
    for bb in backbones:
        for spect in SPECTS:
            for subset in SUBSETS:
                train(
                    dataset="ccmusic-database/chest_falsetto",
                    subset=subset,
                    data_col=spect,
                    label_col="label",
                    backbone=bb,
                    focal_loss=True,
                    full_finetune=True,
                    epoch_num=40,
                )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    erhu_exp()
    bel_exp()
    chest_exp()
