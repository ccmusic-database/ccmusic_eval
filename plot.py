import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "Times New Roman"


def plot_with_values(
    labels: list,
    values: list,
    filename: str = "./MSA_Dataset/msa.pdf",
    aspect_ratio: float = 1.2,
    label_fontsize: int = 24,
    tick_fontsize: int = 22,
):
    plt.figure(figsize=(len(labels) * aspect_ratio, 6))
    bars = plt.bar(labels, values, color="cyan", edgecolor="black")
    plt.xticks(rotation=45, ha="right", fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.ylim(top=1020)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=tick_fontsize,
        )

    plt.ylabel("Frequency", fontsize=label_fontsize)
    os.makedirs("MSA_Dataset", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


s = {
    "intro": 280,
    "verse": 852,
    "pre-chorus": 255,
    "chorus": 943,
    "re-intro": 203,
    "ending": 196,
    "bridge": 137,
    "interlude": 52,
    "post-chorus": 2,
}

# Example usage:
labels = list(s.keys())
values = list(s.values())
plot_with_values(labels, values)
