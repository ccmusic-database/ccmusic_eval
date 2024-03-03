"""
Note: This file only draw the result of using 1 beat deteciton algorithm
"""

import mir_eval
import core
import os
import seaborn as sns
from matplotlib import pyplot as plt


def eval():
    ref_path = os.path.join(core.DATASET_DIR, "labAnnotations")
    est_path = os.path.join(core.DATASET_DIR, "labEstimations")

    file_list = os.listdir(ref_path)
    pairwise = []
    f1_05 = []
    f1_3 = []
    entropy = []
    for i in range(len(file_list)):
        # for i in range(1):
        file = file_list[i]
        print(file)
        ref_file = os.path.join(ref_path, file)
        est_file = os.path.join(est_path, file)

        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_file)

        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_file)

        (ref_intervals, ref_labels) = mir_eval.util.adjust_intervals(
            ref_intervals, ref_labels, t_min=0
        )

        (est_intervals, est_labels) = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, t_min=0, t_max=ref_intervals.max()
        )

        precision, recall, f = mir_eval.segment.pairwise(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        # With 0.5s windowing
        pairwise.append(f)

        P05, R05, F05 = mir_eval.segment.detection(
            ref_intervals, est_intervals, window=0.5
        )
        f1_05.append(F05)
        P3, R3, F3 = mir_eval.segment.detection(ref_intervals, est_intervals, window=3)
        f1_3.append(F3)
        # print(F05)
        # f1.append(F05)
        S_over, S_under, S_F = mir_eval.segment.nce(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        entropy.append(S_F)
    return f1_05, f1_3, pairwise, entropy


def plot(f1_05, f1_3, pairwise, entropy):
    fig, ax = plt.subplots()

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

    ax.legend([box1["boxes"][0]], ["Ellis Beats"], loc="lower right")

    positions = [0.2, 0.4, 0.6, 0.8]  # x-positions of the boxplots
    for pos in positions:
        ax.axhline(y=pos, color="gray", linestyle="--", lw=1)

    positions1 = [2, 6, 10, 14]  # x-positions of the boxplots
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

    plt.savefig(os.path.join(os.getcwd(), "\\segment_results.pdf"))
