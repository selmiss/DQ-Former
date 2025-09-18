import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.font_manager as font_manager

prop = font_manager.FontProperties(fname="./results/Times New Roman.ttf", size=12)
prop_bold = font_manager.FontProperties(fname="./results/Times New Roman - Bold.ttf", size=12)
prop_bold_large = font_manager.FontProperties(fname="./results/Times New Roman - Bold.ttf", size=14)


OUT_DIR = Path("./utils/figures/results/heatmap")

def plot_combined_csv(csv1, csv2, outfile):
    A1 = pd.read_csv(csv1).to_numpy()
    A2 = pd.read_csv(csv2).to_numpy()

    # 横向拼接
    A = np.concatenate([A1, A2], axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["white", "pink", "purple"]  # 浅绿 → 深绿
    cmap = mcolors.LinearSegmentedColormap.from_list("light2darkblue", colors, N=256)
    im = ax.imshow(A, aspect="auto",norm=mcolors.PowerNorm(gamma=0.25, vmin=0, vmax=1), interpolation="nearest", cmap=cmap)
    ax.set_xlabel("Tokens (Fixed 1–8, then Dynamic)", fontproperties=prop)
    ax.set_ylabel("Node Embeddings (Atoms)", fontproperties=prop)
    ax.set_title("Attention Heatmap: Fixed + Dynamic Tokens", fontproperties=prop)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Attention Score", fontproperties=prop)
    for label in ax.get_xticklabels():
        label.set_fontproperties(prop)   # 使用同样的字体
        label.set_fontsize(10)           # 字体大小可调

    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(10)
    for spine in ax.spines.values():
        spine.set_edgecolor("lightblue")   # 改成黑色
        spine.set_linewidth(1.2)       # 线条粗细

    fig.tight_layout()
    outpath = OUT_DIR / outfile
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)

combined_plot = plot_combined_csv(
    OUT_DIR/"fixed_tokens.csv",
    OUT_DIR/"dynamic_tokens.csv",
    "fixed_dynamic_combined_heatmap.svg"
)

print("Saved:", combined_plot)