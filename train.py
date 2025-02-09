import csv
import argparse
import warnings
import pandas as pd
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from plot import np, plot_f1, plot_loss, plot_confusion_matrix
from utils import os, torch, tqdm, to_cuda, save_to_csv
from data import DataLoader, prepare_data, load_data
from model import Net, sp_loss, TRAIN_MODES, compute_f1
from trans_model import t_Net

def eval_model(
    model: Net,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    data_col: str,
    label_col: str,
    learning_rate: float,
    best_valid_f1: float,
    loss_list: list,
    log_dir: str,
):
    with torch.no_grad():
        f1 = []
        for data in tqdm(trainLoader, desc="Batch evaluation on trainset"):
            inputs = to_cuda(data[data_col])
            labels = to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicts = F.sigmoid(outputs)

            frame_f1_mean = compute_f1(predicts, labels)
            f1.append(frame_f1_mean)
        train_f1 = sum(f1) / len(f1)
        print(f"Training F1: {train_f1:.2f}")

        f1 = []
        for data in tqdm(validLoader, desc="Batch evaluation on validset"):
            inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicts = F.sigmoid(outputs)
            
            frame_f1_mean = compute_f1(predicts, labels)
            f1.append(frame_f1_mean)
        valid_f1 = sum(f1) / len(f1)
        print(f"Validation F1 : {valid_f1:.2f}")
    
    train_f1, valid_f1 = train_f1.cpu().numpy(), valid_f1.cpu().numpy()

    save_to_csv(f"{log_dir}/f1.csv", [train_f1, valid_f1, learning_rate])
    with open(f"{log_dir}/loss.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for loss in loss_list:
            writer.writerow([loss])

    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        torch.save(model.state_dict(), f"{log_dir}/save.pt")
        print("Model saved.")

    return best_valid_f1


def test_model(
    backbone: str,
    testLoader: DataLoader,
    # classes: list,
    cls_num: int,
    ori_T: int,
    data_col: str,
    label_col: str,
    log_dir: str,
):
    if 'vit' in backbone or 'swin' in backbone:
        model = t_Net(backbone, 0, cls_num, ori_T, weight_path=f"{log_dir}/save.pt")
    else:
        model = Net(backbone, 0, cls_num, ori_T,  weight_path=f"{log_dir}/save.pt")

    f1 = []
    with torch.no_grad():
        for data in tqdm(testLoader, desc="Batch evaluation on testset"):
            inputs = to_cuda(data[data_col])
            labels: torch.Tensor = to_cuda(data[label_col])
            outputs = model.forward(inputs)
            predicts = F.sigmoid(outputs)
            
            frame_f1_mean = compute_f1(predicts, labels)
            f1.append(frame_f1_mean)
        test_f1 = sum(f1) / len(f1)
        print(f"Test F1 : {test_f1:.2f}")
    return test_f1


def save_log(
    # classes: list,
    cls_num: int,
    # cm: np.ndarray,
    start_time: datetime,
    finish_time: datetime,
    # cls_report: str,
    test_f1: torch.tensor,
    log_dir: str,
    backbone_name: str,
    dataset_name: str,
    data_col: str,
    label_col: str,
    best_train_f1: float,
    best_eval_f1: float,
    train_mode: int,
    batch_size: int,
):
    log = f"""
Backbone       : {backbone_name}
Training mode  : {TRAIN_MODES[train_mode]}
Dataset        : {dataset_name}
Data column    : {data_col}
Label column   : {label_col}
Class num      : {cls_num}
Batch size     : {batch_size}
Start time     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Finish time    : {finish_time.strftime('%Y-%m-%d %H:%M:%S')}
Time cost      : {(finish_time - start_time).seconds}s
Best train f1 : {round(best_train_f1, 2)}%
Best eval f1  : {round(best_eval_f1, 2)}%
"""
    # with open(f"{log_dir}/result.log", "w", encoding="utf-8") as f:
    #     f.write(cls_report + log)

    # # save confusion_matrix
    # np.savetxt(f"{log_dir}/mat.csv", cm, delimiter=",", encoding="utf-8")
    # plot_confusion_matrix(cm, classes, log_dir)
    # print(f"{cls_report}\nConfusion matrix :\n{cm.round(3)}\n{log}")
    with open(f'{log_dir}/test_f1.txt', 'w') as f:
        f.write(str(test_f1.cpu().numpy()))

def save_history(
    log_dir: str,
    testLoader: DataLoader,
    # classes: list,
    cls_num: int, 
    start_time: str,
    dataset: str,
    subset: str,
    data_col: str,
    label_col: str,
    backbone: str,
    imgnet_ver: str,
    train_mode: int,
    batch_size: int,
    ori_T: int
):
    finish_time = datetime.now()
    # cls_report, cm = test_model(
    #     backbone,
    #     testLoader,
    #     classes,
    #     data_col,
    #     label_col,
    #     log_dir,
    # )
    test_f1 = test_model(
       backbone,
        testLoader,
        # classes,
        cls_num,
        ori_T,
        data_col,
        label_col,
        log_dir,
    )
    f1_list = pd.read_csv(f"{log_dir}/f1.csv")
    tra_f1_list = f1_list["tra_f1_list"].tolist()
    val_f1_list = f1_list["val_f1_list"].tolist()
    loss_list = pd.read_csv(f"{log_dir}/loss.csv")["loss_list"].tolist()
    plot_f1(tra_f1_list, val_f1_list, log_dir)
    plot_loss(loss_list, log_dir)
    save_log(
        # classes,
        cls_num,
        # cm,
        start_time,
        finish_time,
        # cls_report,
        test_f1,
        log_dir,
        backbone + (f" - ImageNet {imgnet_ver.upper()}" if train_mode < 2 else ""),
        f"{dataset} - {subset}",
        data_col,
        label_col,
        max(tra_f1_list),
        max(val_f1_list),
        train_mode,
        batch_size,
    )


def train(
    data_col: str, # mel, cqt or chroma
    backbone: str,
    dataset: str = "ccmusic-database/Guzheng_Tech99",
    subset: str = "eval",
    label_col: str = "label",
    train_mode: int = 1,
    imgnet_ver="v1",
    batch_size=4,
    epochs=40,
    iteration=10,
    lr=0.001,
):
    # prepare data, we is for sp_loss
    ds, we = prepare_data(dataset, subset, label_col)
    # init model
    temp = next(iter(ds['train']))

    cls_num = len(temp[label_col])

    original_T = len(temp[label_col][0])
    if 'vit' in backbone or 'swin' in backbone:
        model = t_Net(backbone, train_mode, cls_num, original_T, imgnet_ver)
    else:
        model = Net(backbone, train_mode, cls_num, original_T, imgnet_ver)
    # load data
    traLoader, valLoader, tesLoader = load_data(
        ds,
        data_col,
        label_col,
        model.get_input_size(),
        str(model.model).find("BatchNorm") > 0,
        batch_size=batch_size,
    )

    # loss & optimizer
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
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # start training
    best_eval_f1 = 0.0
    start_time = datetime.now()
    log_dir = f"./logs/{dataset.replace('/', '_')}/{backbone}_{data_col}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(log_dir, exist_ok=True)
    save_to_csv(f"{log_dir}/loss.csv", ["loss_list"])
    save_to_csv(f"{log_dir}/f1.csv", ["tra_f1_list", "val_f1_list", "lr_list"])
    
    print(f"Start training {backbone} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...")


    # loop over the dataset multiple times
    for ep in range(epochs):
        loss_list = []
        running_loss = 0.0
        lr: float = optimizer.param_groups[0]["lr"]
        with tqdm(total=len(traLoader), unit="batch") as pbar:
            for i, data in enumerate(traLoader, 0):
                # get the inputs
                inputs, labels = to_cuda(data[data_col]), to_cuda(data[label_col])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model.forward(inputs)

                loss: torch.Tensor = sp_loss(outputs, labels, we)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # print every 2000 mini-batches
                if i % iteration == iteration - 1:
                    pbar.set_description(
                        f"{dataset.split('/')[1]} {data_col} {backbone}: ep={ep + 1}/{epochs}, lr={lr}, loss={round(running_loss / iteration, 4)}"
                    )
                    loss_list.append(running_loss / iteration)

                running_loss = 0.0
                pbar.update(1)

        best_eval_f1 = eval_model(
            model,
            traLoader,
            valLoader,
            data_col,
            label_col,
            lr,
            best_eval_f1,
            loss_list,
            log_dir,
        )
        scheduler.step(loss.item())

    save_history(log_dir, tesLoader, cls_num, start_time, dataset,
        subset, data_col, label_col, backbone, imgnet_ver, train_mode,
        batch_size, original_T
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--ds", type=str, default="ccmusic-database/Guzheng_Tech99")
    parser.add_argument("--subset", type=str, default="eval")
    parser.add_argument("--data", type=str, default="chroma")
    parser.add_argument("--label", type=str, default="label")
    parser.add_argument("--model", type=str, default="squeezenet1_1")
    parser.add_argument("--imgnet", type=str, default="v1")
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--eps", type=int, default=40)
    args = parser.parse_args()
    train(
        dataset=args.ds,
        subset=args.subset,
        data_col=args.data,
        label_col=args.label,
        backbone=args.model,
        imgnet_ver=args.imgnet,
        train_mode=args.mode,
        batch_size=args.bsz,
        epochs=args.eps,
    )
