import os
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from modelscope.msdatasets import MsDataset
from torchvision.transforms import Compose, Resize, RandomAffine, ToTensor, Normalize


def transform(example_batch, data_column: str, label_column: str, img_size: int):
    compose = Compose(
        [
            Resize([img_size, img_size]),
            RandomAffine(5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    inputs = [compose(x) for x in example_batch[data_column]]
    example_batch[data_column] = inputs
    keys = list(example_batch.keys())
    for key in keys:
        if not (key == data_column or key == label_column):
            del example_batch[key]

    return example_batch


def get_weight(Ytr: np.ndarray):  # (2493, 258, 6)
    mp = Ytr[:].sum(0).sum(0)  # (6,)
    mmp = mp.astype(np.float32) / mp.sum()
    cc = ((mmp.mean() / mmp) * ((1 - mmp) / (1 - mmp.mean()))) ** 0.3
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq


def prepare_data(dataset: str, subset: str, label_col: str):
    print("Preparing & loading data...")
    ds = MsDataset.load(
        dataset,
        subset_name=subset,
        cache_dir="./__pycache__",
    )
    Ytr = []
    for item in tqdm(ds["train"], desc="Loading trainset..."):
        Ytr.append(item[label_col])

    Ytr = np.array(Ytr)
    inverse_feq = get_weight(Ytr.transpose(0, 2, 1))

    return ds, inverse_feq


def load_data(
    ds: MsDataset,
    data_col: str,
    label_col: str,
    input_size: int,
    has_bn: bool,
    shuffle=True,
    batch_size=4,
):
    bs = batch_size
    if has_bn:
        print("The model has bn layer")
        if bs < 2:
            print("Switch batch_size >= 2")
            bs = 2

    trainset = ds["train"].with_transform(
        partial(
            transform,
            data_column=data_col,
            label_column=label_col,
            img_size=input_size,
        )
    )
    validset = ds["validation"].with_transform(
        partial(
            transform,
            data_column=data_col,
            label_column=label_col,
            img_size=input_size,
        )
    )
    testset = ds["test"].with_transform(
        partial(
            transform,
            data_column=data_col,
            label_column=label_col,
            img_size=input_size,
        )
    )
    num_workers = os.cpu_count() // 2
    traLoader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=has_bn,
    )
    valLoader = DataLoader(
        validset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=has_bn,
    )
    tesLoader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=has_bn,
    )

    return traLoader, valLoader, tesLoader
