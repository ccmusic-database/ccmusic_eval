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


def prepare_data(dataset: str, subset: str, label_col: str, focal_loss: bool):
    print("Preparing data...")
    ds = MsDataset.load(
        dataset,
        subset_name=subset,
        cache_dir="./__pycache__",
    )
    try:
        classes = ds["test"]._hf_ds.features[label_col].names
    except AttributeError:
        classes = ds["test"].features[label_col].names
    except KeyError:
        print("Ensure the selected dataset has splits: train, validation and test")
        exit()

    num_samples = []

    if focal_loss:
        each_nums = {k: 0 for k in classes}
        for item in tqdm(ds["train"], desc="Statistics by category for focal loss..."):
            each_nums[classes[item[label_col]]] += 1

        num_samples = list(each_nums.values())

    print("The data is prepared.")
    return ds, classes, num_samples


def load_data(
    ds: MsDataset,
    data_col: str,
    label_col: str,
    input_size: int,
    has_bn: bool,
    shuffle=True,
    batch_size=4,
    num_workers=2,
):
    print("Loadeding data...")
    bs = batch_size
    try:
        ds_train = ds["train"]._hf_ds
        ds_valid = ds["validation"]._hf_ds
        ds_test = ds["test"]._hf_ds
    except AttributeError:
        ds_train = ds["train"]
        ds_valid = ds["validation"]
        ds_test = ds["test"]
    except KeyError:
        print("Ensure the selected dataset has splits: train, validation and test")
        exit()

    if has_bn:
        print("The model has bn layer")
        if bs < 2:
            print("Switch batch_size >= 2")
            bs = 2

    trainset = ds_train.with_transform(
        partial(
            transform,
            data_column=data_col,
            label_column=label_col,
            img_size=input_size,
        )
    )
    validset = ds_valid.with_transform(
        partial(
            transform,
            data_column=data_col,
            label_column=label_col,
            img_size=input_size,
        )
    )
    testset = ds_test.with_transform(
        partial(
            transform,
            data_column=data_col,
            label_column=label_col,
            img_size=input_size,
        )
    )

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
    print("The data is loaded.")

    return traLoader, valLoader, tesLoader
