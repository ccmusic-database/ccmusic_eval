import os
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
    inputs = [compose(x.convert("RGB")) for x in example_batch[data_column]]
    example_batch[data_column] = inputs
    keys = list(example_batch.keys())
    for key in keys:
        if not (key == data_column or key == label_column):
            del example_batch[key]

    return example_batch


def prepare_data(dataset: str, subset: str, label_col: str, use_wce: bool):
    print("Preparing & loading data...")
    ds = MsDataset.load(
        dataset,
        subset_name=subset,
        cache_dir="./__pycache__",
    )
    classes = ds["test"].features[label_col].names
    num_samples = []
    if use_wce:
        each_nums = {k: 0 for k in classes}
        for item in tqdm(ds["train"], desc="Statistics by category for WCE loss"):
            each_nums[classes[item[label_col]]] += 1

        num_samples = list(each_nums.values())

    return ds, classes, num_samples


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
