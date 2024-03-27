"""
Note: This file only draw the result of using 1 beat deteciton algorithm
"""

import os
import core
import zipfile
import mir_eval
from tqdm import tqdm
from matplotlib import pyplot as plt

BEATS = [
    "librosa",
    "madmom1",
    "madmom2",
]


def extract_zip(zip_file_path, extract_folder):
    """
    解压 ZIP 文件到指定文件夹。

    参数：
    - zip_file_path: 要解压的 ZIP 文件的路径
    - extract_folder: 解压目标文件夹的路径
    """
    # 确保 ZIP 文件存在
    if not os.path.exists(zip_file_path):
        print(f"Error: ZIP file '{zip_file_path}' does not exist.")
        return
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    # 打开 ZIP 文件
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # 解压缩到目标文件夹
        zip_ref.extractall(extract_folder)

    print(f"Extraction complete. Files extracted to: {extract_folder}")


def plot(f1_05, f1_3, pairwise, entropy, beatype: str):
    _, ax = plt.subplots()
    c1 = "turquoise"
    box1 = ax.boxplot(
        (f1_05, f1_3, pairwise, entropy),
        positions=[1, 5, 9, 13],
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor=c1, color="purple"),
        capprops=dict(color=c1),
        whiskerprops=dict(color=c1),
        flierprops=dict(color=c1, markeredgecolor=c1),
        medianprops=dict(color=c1),
    )

    for item in ["boxes", "whiskers", "fliers", "medians", "caps"]:
        plt.setp(box1[item], color=c1)

    plt.setp(box1["boxes"], facecolor=c1)
    plt.setp(box1["fliers"], markeredgecolor=c1)
    ax.legend([box1["boxes"][0]], [beatype], loc="lower right")
    positions = [0.2, 0.4, 0.6, 0.8]  # x-positions of the boxplots
    positions1 = [2, 6, 10, 14]  # x-positions of the boxplots
    for pos in positions:
        ax.axhline(y=pos, color="gray", linestyle="--", lw=1)

    for pos in positions1:
        ax.axvline(x=pos, color="gray", linestyle="--", lw=1)

    plt.xticks(
        [2, 6, 10, 14],
        ["Hit Rate@0.5", "Hit Rate@3", "Pairwise Clust.", "Entropy Scores"],
    )
    plt.xlim(0, 16)
    plt.ylim(-0.05, 1)
    plt.ylabel("F-measures")
    plt.tight_layout()
    plt.savefig("./MSA_dataset/segment_results.pdf")


def plot_all(librosa_df: dict, kwski_df: dict, kreb_df: dict):
    _, ax = plt.subplots()
    c1 = "turquoise"
    box1 = ax.boxplot(
        (
            librosa_df["HitRate_0.5F"],
            librosa_df["HitRate_3F"],
            librosa_df["PWF"],
            librosa_df["Sf"],
        ),
        positions=[1, 5, 9, 13],
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor=c1, color="purple"),
        capprops=dict(color=c1),
        whiskerprops=dict(color=c1),
        flierprops=dict(color=c1, markeredgecolor=c1),
        medianprops=dict(color=c1),
    )
    c2 = "orchid"
    box2 = ax.boxplot(
        (
            kwski_df["HitRate_0.5F"],
            kwski_df["HitRate_3F"],
            kwski_df["PWF"],
            kwski_df["Sf"],
        ),
        positions=[2, 6, 10, 14],
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor=c2, color="purple"),
        capprops=dict(color=c2),
        whiskerprops=dict(color=c2),
        flierprops=dict(color=c2, markeredgecolor=c2),
        medianprops=dict(color=c2),
    )
    c3 = "purple"
    box3 = ax.boxplot(
        (kreb_df["HitRate_0.5F"], kreb_df["HitRate_3F"], kreb_df["PWF"], kreb_df["Sf"]),
        positions=[3, 7, 11, 15],
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor=c3, color=c3),
        capprops=dict(color=c3),
        whiskerprops=dict(color=c3),
        flierprops=dict(color=c3, markeredgecolor=c3),
        medianprops=dict(color=c3),
    )
    for item in ["boxes", "whiskers", "fliers", "medians", "caps"]:
        plt.setp(box3[item], color=c3)

    plt.setp(box3["boxes"], facecolor=c3)
    plt.setp(box3["fliers"], markeredgecolor=c3)

    ax.legend(
        [box1["boxes"][0], box2["boxes"][0], box3["boxes"][0]],
        ["Ellis Beats", "Korzeniowski Beats", "Krebs Beats"],
        loc="lower right",
    )

    plt.xticks(
        [2, 6, 10, 14],
        ["Hit Rate@0.5", "Hit Rate@3", "Pairwise Clust.", "Entropy Scores"],
    )
    plt.xlim(0, 16)
    plt.ylim(-0.05, 1)
    plt.ylabel("F-measures")
    plt.tight_layout()
    outpath = "./MSA_dataset/segment_results.pdf"
    plt.savefig(outpath)
    print(f"The segment results have been saved at {outpath}.")


def eval(est_dir: str):
    f1_05, f1_3, pairwise, entropy = [], [], [], []
    ref_path = os.path.join(core.DATASET_DIR, "labAnnotations")
    est_path = os.path.join(core.DATASET_DIR, est_dir)
    file_list = os.listdir(ref_path)
    for i in tqdm(range(len(file_list)), desc=f"Processing files in {est_dir}..."):
        file = file_list[i]
        ref_file = os.path.join(ref_path, file)
        est_file = os.path.join(est_path, file)
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_file)
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_file)
        ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
            ref_intervals, ref_labels, t_min=0
        )
        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, t_min=0, t_max=ref_intervals.max()
        )
        _, _, f = mir_eval.segment.pairwise(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        # With 0.5s windowing
        pairwise.append(f)
        _, _, F05 = mir_eval.segment.detection(ref_intervals, est_intervals, window=0.5)
        f1_05.append(F05)
        _, _, F3 = mir_eval.segment.detection(ref_intervals, est_intervals, window=3)
        f1_3.append(F3)
        _, _, S_F = mir_eval.segment.nce(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        entropy.append(S_F)

    return f1_05, f1_3, pairwise, entropy


if __name__ == "__main__":
    extract_zip("MSA_Dataset.zip", "./")
    results = []
    for beat in BEATS:
        f1_05, f1_3, pairwise, entropy = eval(f"labEstimations_{beat}")
        results.append(
            {"HitRate_0.5F": f1_05, "HitRate_3F": f1_3, "PWF": pairwise, "Sf": entropy}
        )

    plot_all(results[0], results[1], results[2])
