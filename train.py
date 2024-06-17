import csv
import argparse
import warnings
import torch.utils.data
import torch.optim as optim
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from plot import np, plot_acc, plot_loss, plot_confusion_matrix
from data import DataLoader, prepare_data, load_data
from utils import torch, tqdm, to_cuda, save_to_csv
from focalLoss import FocalLoss
from model import os, nn, Net


def eval_model(
    model: Net,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    data_col: str,
    label_col: str,
    learning_rate: float,
    best_valid_acc: float,
    loss_list: list,
    log_dir: str,
):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in tqdm(trainLoader, desc="Batch evaluation on trainset"):
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

        train_acc = 100.0 * accuracy_score(y_true, y_pred)
        print(f"Training accuracy : {round(train_acc, 2)}%")

        y_true, y_pred = [], []
        for data in tqdm(validLoader, desc="Batch evaluation on validset"):
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

        valid_acc = 100.0 * accuracy_score(y_true, y_pred)
        print(f"Validation accuracy : {round(valid_acc, 2)}%")

    save_to_csv(log_dir + "/acc.csv", [train_acc, valid_acc, learning_rate])
    with open(log_dir + "/loss.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for loss in loss_list:
            writer.writerow([loss])

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), log_dir + "/save.pt")
        print("Model saved.")

    return best_valid_acc


def test_model(
    backbone: str,
    testLoader: DataLoader,
    classes: list,
    data_col: str,
    label_col: str,
    log_dir: str,
):
    model = Net(backbone, len(classes), False, weight_path=f"{log_dir}/save.pt")
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in tqdm(testLoader, desc="Batch evaluation on testset"):
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
    start_time: datetime,
    finish_time: datetime,
    cls_report: str,
    log_dir: str,
    backbone: str,
    dataset: str,
    data_col: str,
    label_col: str,
    focal_loss: str,
    best_train_acc: float,
    best_eval_acc: float,
    full_finetune: bool,
):
    log = f"""
Backbone       : {backbone}
Dataset        : {dataset}
Data column    : {data_col}
Label column   : {label_col}
Class num      : {len(classes)}
Start time     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Finish time    : {finish_time.strftime('%Y-%m-%d %H:%M:%S')}
Time cost      : {(finish_time - start_time).seconds}s
Full finetune  : {full_finetune}
Use focal loss : {focal_loss}
Best train acc : {round(best_train_acc, 2)}%
Best eval acc  : {round(best_eval_acc, 2)}%
"""

    with open(f"{log_dir}/result.log", "w", encoding="utf-8") as f:
        f.write(cls_report + log)

    # save confusion_matrix
    np.savetxt(f"{log_dir}/mat.csv", cm, delimiter=",", encoding="utf-8")
    plot_confusion_matrix(cm, classes, log_dir)
    print(f"{cls_report}\nConfusion matrix :\n{cm.round(3)}\n{log}")


def save_history(
    log_dir: str,
    testLoader: DataLoader,
    classes: list,
    start_time: str,
    finish_time: str,
    dataset: str,
    data_col: str,
    label_col: str,
    backbone: str,
    focal_loss: str,
    full_finetune: bool,
):
    cls_report, cm = test_model(
        backbone, testLoader, classes, data_col, label_col, log_dir
    )

    acc_list = pd.read_csv(log_dir + "/acc.csv")
    tra_acc_list = acc_list["tra_acc_list"].tolist()
    val_acc_list = acc_list["val_acc_list"].tolist()
    loss_list = pd.read_csv(log_dir + "/loss.csv")["loss_list"].tolist()

    plot_acc(tra_acc_list, val_acc_list, log_dir)
    plot_loss(loss_list, log_dir)
    save_log(
        classes,
        cm,
        start_time,
        finish_time,
        cls_report,
        log_dir,
        backbone,
        dataset,
        data_col,
        label_col,
        focal_loss,
        max(tra_acc_list),
        max(val_acc_list),
        full_finetune,
    )


def train(
    dataset: str,
    subset: str,
    data_col: str,
    label_col: str,
    backbone: str,
    focal_loss: bool,
    full_finetune: bool,
    epoch_num=40,
    iteration=10,
    lr=0.001,
):
    # prepare data
    ds, classes, num_samples = prepare_data(dataset, subset, label_col, focal_loss)

    # init model
    model = Net(backbone, len(classes), full_finetune)

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

    # start training
    start_time = datetime.now()
    log_dir = f"./logs/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Start tuning {backbone} at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ...")
    save_to_csv(log_dir + "/acc.csv", ["tra_acc_list", "val_acc_list", "lr_list"])
    save_to_csv(log_dir + "/loss.csv", ["loss_list"])

    best_eval_acc = 0.0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        lr: float = optimizer.param_groups[0]["lr"]
        running_loss = 0.0
        loss_list = []
        with tqdm(total=len(traLoader), unit="batch") as pbar:
            for i, data in enumerate(traLoader, 0):
                # get the inputs
                inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model.forward(inputs)
                loss: torch.Tensor = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # print every 2000 mini-batches
                if i % iteration == iteration - 1:
                    pbar.set_description(
                        "epoch=%d/%d, lr=%.4f, loss=%.4f"
                        % (
                            epoch + 1,
                            epoch_num,
                            lr,
                            running_loss / iteration,
                        )
                    )
                    loss_list.append(running_loss / iteration)

                running_loss = 0.0
                pbar.update(1)

        best_eval_acc = eval_model(
            model,
            traLoader,
            valLoader,
            data_col,
            label_col,
            lr,
            best_eval_acc,
            loss_list,
            log_dir,
        )
        scheduler.step(loss.item())

    save_history(
        log_dir,
        tesLoader,
        classes,
        start_time,
        datetime.now(),
        f"{dataset} - {subset}",
        data_col,
        label_col,
        backbone,
        focal_loss,
        full_finetune,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--dataset", type=str, default="ccmusic/chest_falsetto")
    parser.add_argument("--subset", type=str, default="eval")
    parser.add_argument("--data", type=str, default="cqt")
    parser.add_argument("--label", type=str, default="singing_method")
    parser.add_argument("--backbone", type=str, default="squeezenet1_1")
    parser.add_argument("--focalloss", type=bool, default=True)
    parser.add_argument("--fullfinetune", type=bool, default=False)
    args = parser.parse_args()

    train(
        dataset=args.dataset,  # dataset on modelscope
        subset=args.subset,
        data_col=args.data,
        label_col=args.label,
        backbone=args.backbone,
        focal_loss=args.focalloss,
        full_finetune=args.fullfinetune,
        epoch_num=2,  # 2 epochs only for test
    )
