import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss

plt.rcParams["font.sans-serif"] = "Times New Roman"


def show_point(max_id, list):
    show_max = f"({max_id + 1}, {round(list[max_id], 2)})"
    plt.annotate(
        show_max,
        xytext=(max_id + 1, list[max_id]),
        xy=(max_id + 1, list[max_id]),
        fontsize=6,
    )


def smooth(y):
    if 95 <= len(y):
        return ss.savgol_filter(y, 95, 3)

    return y


def plot_acc(tra_acc_list, val_acc_list, save_path):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.title("Accuracy of training and validation", fontweight="bold")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1 + max1, y1[max1], "r-o")
    plt.plot(1 + max2, y2[max2], "r-o")
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()
    plt.savefig(save_path + "/acc.pdf", bbox_inches="tight")
    plt.close()


def plot_loss(loss_list, save_path):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title("Loss curve", fontweight="bold")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.plot(x_loss, smooth(loss_list))
    plt.savefig(save_path + "/loss.pdf", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray, labels_name: list, save_path: str, title="Confusion matrix"
):
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalized
    # Display an image on a specific window
    plt.imshow(cm, interpolation="nearest")
    plt.title(title, fontweight="bold")  # image caption
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    # print the labels on the x-axis coordinates
    plt.xticks(num_local, labels_name, rotation=90)
    # print the label on the y-axis coordinate
    plt.yticks(num_local, labels_name)
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.tight_layout()
    plt.savefig(save_path + "/mat.pdf", bbox_inches="tight")
    plt.close()
