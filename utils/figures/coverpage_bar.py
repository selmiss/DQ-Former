import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
# 任务和指标
tasks = ["Pampa", "BBBP", "Structure", "Source", "Property", "Application"]
metrics = ["(EM)", "(Pass@1)", "(EM)", "(Pass@1)", "(Percentile)", "(Resolved)"]

# 模型
models = [
    "EDT-Former-8.3B",
    "GPT-5",
    "Galactica-6.7B",
    "Mol-Instructions-8B",
    "Mol-Llama-3.1-8B",
    "3D-MoLM-8B"
]

# 每个模型的得分
scores = np.array([
    [82.34, 72.48, 74.55, 72.93, 50.71, 48.58, ],  # 模型1
    [51.35, 62.17, 33.28, 33.49, 32.41, 26.26, ],  # 模型2
    [55.37, 59.31, 32.35, 41.92, 31.05, 28.21, ],  # 模型3
    [57.39, 53.29, 72.79, 70.82, 43.08, 41.22, ],  # 模型4
    [67.15, 56.64, 73.16, 70.22, 45.70, 46.18, ],  # 模型5
    [54.50, 51.20, 73.17, 70.50, 44.79, 44.19, ],  # 模型6
])

# 柱状图参数
bar_width = 0.13
x = np.arange(len(tasks))
colors = ["dodgerblue", "deepskyblue", "lightgrey", "tan", "khaki", "wheat"]
prop = font_manager.FontProperties(fname="./utils/figures/results/Times New Roman.ttf", size=18)
prop_bold = font_manager.FontProperties(fname="./utils/figures/results/Times New Roman - Bold.ttf")
print("Loaded font:", prop.get_name())
rcParams["font.family"] = prop.get_name()

# plt.rcParams["font.family"] = prop.get_name()




plt.figure(figsize=(12, 7.5))

plt.grid(axis="y", color="lightgray", linestyle="-", linewidth=0.5)

for i, model in enumerate(models):
    plt.bar(
        x + i * bar_width,
        scores[i],
        width=bar_width,
        label=model,
        color=colors[i],
        edgecolor="white",
        linewidth=0.5,
        hatch="//" if "EDT-Former-8.3B" in model else None  # 给 DeepSeek-V3 加斜线
    )
    # 在柱子顶部加数值
    for xi, yi in zip(x + i * bar_width, scores[i]):
        if "EDT-Former-8.3B" in model:
            plt.text(xi, yi + 0.4, f"{yi:.1f}", ha="center", va="bottom", fontsize=15, fontweight="bold", fontproperties=prop_bold)
        else:
            plt.text(xi, yi + 0.4, f"{yi:.1f}", ha="center", va="bottom", fontsize=9, fontproperties=prop)

# 坐标轴 & 样式

plt.xticks(x + bar_width * (len(models)-1)/2, [f"{t}" for t, m in zip(tasks, metrics)], fontproperties=prop, fontsize=18)
# plt.ylim(0, 90)
plt.ylabel("Accuracy (%)", fontproperties=prop, fontsize=18)
plt.ylim(20, 90)



plt.legend(
    loc="upper right",      # 图例放在图形上方
    # bbox_to_anchor=(0.5, 1.18),  # (x, y)，y>1 表示在图外上方
    ncol=1,                  # 横向排列，3 列
    frameon=True,            # 显示边框
    fontsize=20,              # 字体大小
    edgecolor="lightblue",        # 边框颜色
    prop=prop,
    columnspacing=5
)
# 设置边框颜色和宽度
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(18)
for spine in ax.spines.values():
    spine.set_edgecolor("lightblue")
    spine.set_linewidth(1.0)

plt.tight_layout()

# 保存图片
os.makedirs("./utils/figures/results", exist_ok=True)
plt.savefig("./utils/figures/results/coverpage_v4.pdf", dpi=600, bbox_inches="tight")  # 也可以保存为 PDF 矢量图
plt.close()
