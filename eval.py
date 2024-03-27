"""
Note: This file only draw the result of using 1 beat deteciton algorithm
"""

import os
import core
import mir_eval
from matplotlib import pyplot as plt

BEATS = {
    "librosa": "Ellis Beats",
    "madmom1": "Korzeniowski Beats",
    "madmom2": "Krebs Beats",
}


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


def eval(est_dir: str, pairwise: list, f1_05: list, f1_3: list, entropy: list):
    ref_path = os.path.join(core.DATASET_DIR, "labAnnotations")
    est_path = os.path.join(core.DATASET_DIR, est_dir)
    file_list = os.listdir(ref_path)
    for i in range(len(file_list)):
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


if __name__ == "__main__":
    f1_05, f1_3, pairwise, entropy = [], [], [], []
    for beat in BEATS.keys():
        f1_05, f1_3, pairwise, entropy = eval(
            f"labEstimations_{beat}",
            f1_05,
            f1_3,
            pairwise,
            entropy,
        )
        plot(f1_05, f1_3, pairwise, entropy, BEATS[beat])


# fig, ax = plt.subplots()

# c1 = "turquoise"
# box1 = ax.boxplot((librosa_df["HitRate_0.5F"],
#              librosa_df["HitRate_3F"],
#              librosa_df["PWF"],
#              librosa_df["Sf"]),
#             positions=[1, 5, 9, 13],
#             notch=True, patch_artist=True,
#             boxprops=dict(facecolor=c1, color="purple"),
#             capprops=dict(color=c1),
#             whiskerprops=dict(color=c1),
#             flierprops=dict(color=c1, markeredgecolor=c1),
#             medianprops=dict(color=c1))

# c2 = "orchid"
# box2 = ax.boxplot((kwski_df["HitRate_0.5F"],
#              kwski_df["HitRate_3F"],
#              kwski_df["PWF"],
#              kwski_df["Sf"]),
#             positions=[2, 6, 10, 14],
#             notch=True, patch_artist=True,
#             boxprops=dict(facecolor=c2, color="purple"),
#             capprops=dict(color=c2),
#             whiskerprops=dict(color=c2),
#             flierprops=dict(color=c2, markeredgecolor=c2),
#             medianprops=dict(color=c2))

# c3 = "purple"
# box3 = ax.boxplot((kreb_df["HitRate_0.5F"],
#              kreb_df["HitRate_3F"],
#              kreb_df["PWF"],
#              kreb_df["Sf"]),
#             positions=[3, 7, 11, 15],
#             notch=True, patch_artist=True,
#             boxprops=dict(facecolor=c3, color=c3),
#             capprops=dict(color=c3),
#             whiskerprops=dict(color=c3),
#             flierprops=dict(color=c3, markeredgecolor=c3),
#             medianprops=dict(color=c3))
# for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
#     plt.setp(box3[item], color=c3)
# plt.setp(box3["boxes"], facecolor=c3)
# plt.setp(box3["fliers"], markeredgecolor=c3)

# ax.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0]],
#           ['Ellis Beats', 'Korzeniowski Beats', 'Krebs Beats'],
#           loc='lower right')

# plt.xticks([2, 6, 10, 14], ["Hit Rate@0.5", "Hit Rate@3", "Pairwise Clust.", "Entropy Scores"])
# plt.xlim(0, 16)
# plt.ylim(-0.05, 1)

# plt.ylabel("F-measures")
# plt.tight_layout()
# plt.savefig("../paper/fig")
