import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss

plt.rcParams["font.sans-serif"] = "Times New Roman"


def show_point(max_id: int, data_list: list):
    show_max = f"({max_id + 1}, {round(data_list[max_id], 2)})"
    plt.annotate(
        show_max,
        xytext=(max_id + 1, data_list[max_id]),
        xy=(max_id + 1, data_list[max_id]),
        fontsize=6,
    )


def smooth(y: list):
    if 95 <= len(y):
        return ss.savgol_filter(y, 95, 3)

    return y


def plot_acc(tra_acc_list: list, val_acc_list: list, save_path: str):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)
    plt.title("Accuracy of training and validation", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1 + max1, y1[max1], "r-o")
    plt.plot(1 + max2, y2[max2], "r-o")
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()
    plt.savefig(save_path + "/acc.pdf", bbox_inches="tight")
    plt.close()


def plot_loss(loss_list: list, save_path: str):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title("Loss curve", fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(x_loss, smooth(loss_list))
    plt.savefig(save_path + "/loss.pdf", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels_name: list,
    save_path: str,
    title="Confusion matrix",
):
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalized
    plt.imshow(cm, interpolation="nearest", cmap="Blues")  # 使用 'Blues' colormap
    plt.title(title, fontweight="bold")  # 图像标题
    plt.colorbar()
    count = len(labels_name)
    num_local = np.array(range(count))
    if count <= 20:
        # 在色块上添加数值
        for i in range(count):
            for j in range(count):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], ".2f"),
                    horizontalalignment="center",
                    color="black" if cm[i, j] <= 0.5 else "white",
                )  # 根据色块亮度选择文本颜色
        # 在x轴坐标上打印标签
        plt.xticks(num_local, labels_name, rotation=45)
        # 在y轴坐标上打印标签
        plt.yticks(num_local, labels_name)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path + "/mat.pdf", bbox_inches="tight")
    plt.close()
