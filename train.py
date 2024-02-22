import os
import csv
import torch
import argparse
import warnings
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import Net
from datetime import datetime
from functools import partial
from focalLoss import FocalLoss
from torch.utils.data import DataLoader
from modelscope.msdatasets import MsDataset
from torchvision.transforms import Compose, Resize, RandomAffine, ToTensor, Normalize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from plot import save_acc, save_loss, save_confusion_matrix
from utils import time_stamp, to_cuda


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
    ds = MsDataset.load(dataset, subset_name=subset)
    classes = ds["test"]._hf_ds.features[label_col].names
    num_samples = []

    if focal_loss:
        each_nums = {k: 0 for k in classes}
        for item in ds["train"]:
            each_nums[classes[item[label_col]]] += 1

        num_samples = list(each_nums.values())

    print("Data prepared.")
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
    ds_train = ds["train"]._hf_ds
    ds_valid = ds["validation"]._hf_ds
    ds_test = ds["test"]._hf_ds

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
    print("Data loaded.")

    return traLoader, valLoader, tesLoader


def eval_model_train(
    model: Net,
    trainLoader: DataLoader,
    tra_acc_list: list,
    data_col: str,
    label_col: str,
):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in trainLoader:
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print(f"Training acc   : {str(round(acc, 2))}%")
    tra_acc_list.append(acc)


def eval_model_valid(
    model: Net,
    validLoader: DataLoader,
    val_acc_list: list,
    data_col: str,
    label_col: str,
):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in validLoader:
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print(f"Validation acc : {str(round(acc, 2))}%")
    val_acc_list.append(acc)


def eval_model_test(
    model: Net, testLoader: DataLoader, classes: list, data_col: str, label_col: str
):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    cm = confusion_matrix(y_true, y_pred, normalize="all")

    return report, cm


def save_log(
    classes: list,
    cm: np.ndarray,
    start_time: str,
    finish_time: str,
    cls_report: str,
    log_dir: str,
    backbone: str,
    data_col: str,
    focal_loss: str,
    full_finetune: bool,
):
    log = f"""
Class num     : {len(classes)}
Backbone      : {backbone}
Data column   : {data_col}
Start time    : {time_stamp(start_time)}
Finish time   : {time_stamp(finish_time)}
Time cost     : {str((finish_time - start_time).seconds)}s
Full finetune : {str(full_finetune)}
Focal loss    : {focal_loss}"""

    with open(f"{log_dir}/result.log", "w", encoding="utf-8") as f:
        f.write(cls_report + log)
    f.close()

    # save confusion_matrix
    np.savetxt(f"{log_dir}/mat.csv", cm, delimiter=",")
    save_confusion_matrix(cm, classes, log_dir)
    print(f"{cls_report}\nConfusion matrix :\n{str(cm.round(3))}\n{log}")


def save_history(
    model: Net,
    tra_acc_list: list,
    val_acc_list: list,
    loss_list: list,
    lr_list: list,
    classes: list,
    cm: np.ndarray,
    cls_report: str,
    start_time: str,
    finish_time: str,
    dataset: str,
    data_col: str,
    backbone: str,
    focal_loss: str,
    full_finetune: bool,
):
    results_dir = f"./logs/{dataset.replace('/', '_')}"
    log_dir = f"{results_dir}/{backbone}_{data_col}_{len(classes)}cls_{time_stamp()}"
    os.makedirs(log_dir, exist_ok=True)

    acc_len = len(tra_acc_list)
    with open(f"{log_dir}/acc.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tra_acc_list", "val_acc_list", "lr_list"])
        for i in range(acc_len):
            writer.writerow([tra_acc_list[i], val_acc_list[i], lr_list[i]])

    with open(f"{log_dir}/loss.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["loss_list"])
        for loss in loss_list:
            writer.writerow([loss])

    torch.save(model.state_dict(), f"{log_dir}/save.pt")
    print("Model saved.")

    save_acc(tra_acc_list, val_acc_list, log_dir)
    save_loss(loss_list, log_dir)
    save_log(
        classes,
        cm,
        start_time,
        finish_time,
        cls_report,
        log_dir,
        backbone,
        data_col,
        focal_loss,
        full_finetune,
    )


def train(
    dataset: str,
    subset: str,
    data_col: str,
    label_col: str,
    backbone: str,
    pretrain: str,
    focal_loss: bool,
    full_finetune: bool,
    epoch_num=40,
    iteration=10,
    lr=0.001,
):
    # prepare data
    ds, classes, num_samples = prepare_data(dataset, subset, label_col, focal_loss)

    # init model
    model = Net(backbone, pretrain, len(classes), full_finetune)

    # load data
    traLoader, valLoader, tesLoader = load_data(
        ds,
        data_col,
        label_col,
        model.get_input_size(),
        str(model.model).find("BatchNorm") > 0,
    )

    # loss & optimizer
    criterion = FocalLoss(num_samples) if focal_loss else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        verbose=True,
        threshold=lr,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
    )

    # gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        criterion = criterion.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # train
    start_time = datetime.now()
    print(f"Start training [{backbone}] at {time_stamp(start_time)} ...")
    tra_acc_list, val_acc_list, loss_list, lr_list = [], [], [], []
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        lr_str = optimizer.param_groups[0]["lr"]
        lr_list.append(lr_str)
        print(
            f" Epoch {epoch + 1}/{epoch_num} ".center(40, "-"),
            f"\nLearning rate: {lr_str}",
        )
        running_loss = 0.0
        for i, data in enumerate(traLoader, 0):
            # get the inputs
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print every 2000 mini-batches
            if i % iteration == iteration - 1:
                print(
                    "[%d, %5d] loss: %.4f"
                    % (epoch + 1, i + 1, running_loss / iteration)
                )
                loss_list.append(running_loss / iteration)

            running_loss = 0.0

        eval_model_train(model, traLoader, tra_acc_list, data_col, label_col)
        eval_model_valid(model, valLoader, val_acc_list, data_col, label_col)
        scheduler.step(loss.item())

    finish_time = datetime.now()
    cls_report, cm = eval_model_test(model, tesLoader, classes, data_col, label_col)
    save_history(
        model,
        tra_acc_list,
        val_acc_list,
        loss_list,
        lr_list,
        classes,
        cm,
        cls_report,
        start_time,
        finish_time,
        dataset,
        data_col,
        backbone,
        focal_loss,
        full_finetune,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--dataset", type=str, default="ccmusic/chest_falsetto")
    parser.add_argument("--subset", type=str, default="default")
    parser.add_argument("--data", type=str, default="cqt")
    parser.add_argument("--label", type=str, default="label")
    parser.add_argument("--backbone", type=str, default="squeezenet1_1")
    parser.add_argument("--pretrain", type=str, default="ImageNet1k_v1")
    parser.add_argument("--focalloss", type=bool, default=True)
    parser.add_argument("--fullfinetune", type=bool, default=True)
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        subset=args.subset,
        data_col=args.data,
        label_col=args.label,
        backbone=args.backbone,
        pretrain=args.pretrain,
        focal_loss=args.focalloss,
        full_finetune=args.fullfinetune,
    )
