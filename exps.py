import warnings
from train import train

SPECTS = ["mel", "cqt", "chroma"]


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
            train(
                dataset="ccmusic-database/erhu_playing_tech",
                subset="eval",
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
            train(
                dataset="ccmusic-database/bel_canto",
                subset="eval",
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
            train(
                dataset="ccmusic-database/chest_falsetto",
                subset="eval",
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
