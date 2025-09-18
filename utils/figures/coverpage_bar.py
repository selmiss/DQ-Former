import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
# 任务和指标
tasks = ["Structure", "Source", "Property", "Application", "Pampa", "BBBP"]
metrics = ["(EM)", "(Pass@1)", "(EM)", "(Pass@1)", "(Percentile)", "(Resolved)"]

# 模型
models = [
    "DyQ-Former-8B",
    "GPT-5",
    "Galactica-6.7B",
    "Mol-Instructions-7B",
    "Mol-Llama-8B",
    "3D-MolLM-8B"
]

# 每个模型的得分
scores = np.array([
    [77.87, 75.79, 50.20, 48.58, 82.31, 75.00],  # 模型1
    [33.28, 33.49, 32.41, 26.26, 51.35, 62.20],  # 模型2
    [32.35, 41.92, 31.05, 28.21, 55.37, 59.31],  # 模型3
    [75.93, 73.96, 46.22, 44.36, 53.29, 54.55],  # 模型4
    [75.33, 73.20, 45.26, 45.71, 66.80, 58.06],  # 模型5
    [69.64, 68.29, 43.19, 43.81, 53.93, 50.90],  # 模型6
])

# 柱状图参数
bar_width = 0.13
x = np.arange(len(tasks))
colors = ["dodgerblue", "deepskyblue", "lightgrey", "tan", "khaki", "wheat"]
prop = font_manager.FontProperties(fname="./results/Times New Roman.ttf", size=12)
prop_bold = font_manager.FontProperties(fname="./results/Times New Roman - Bold.ttf")
print("Loaded font:", prop.get_name())
rcParams["font.family"] = prop.get_name()

# plt.rcParams["font.family"] = prop.get_name()




plt.figure(figsize=(12, 7))
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
        hatch="//" if "DyQ-Former" in model else None  # 给 DeepSeek-V3 加斜线
    )
    # 在柱子顶部加数值
    for xi, yi in zip(x + i * bar_width, scores[i]):
        if "DyQ-Former" in model:
            plt.text(xi, yi + 0.4, f"{yi:.1f}", ha="center", va="bottom", fontsize=14, fontweight="bold", fontproperties=prop_bold)
        else:
            plt.text(xi, yi + 0.4, f"{yi:.1f}", ha="center", va="bottom", fontsize=10, fontproperties=prop)

# 坐标轴 & 样式

plt.xticks(x + bar_width * (len(models)-1)/2, [f"{t}" for t, m in zip(tasks, metrics)], fontproperties=prop, fontsize=18)
# plt.ylim(0, 90)
plt.ylabel("Accuracy (%)", fontproperties=prop, fontsize=18)


plt.legend(
    loc="upper center",      # 图例放在图形上方
    bbox_to_anchor=(0.5, 1.07),  # (x, y)，y>1 表示在图外上方
    ncol=6,                  # 横向排列，3 列
    frameon=True,            # 显示边框
    # fontsize=20,              # 字体大小
    edgecolor="lightblue",        # 边框颜色
    prop=prop
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
os.makedirs("./results", exist_ok=True)
plt.savefig("./results/coverpage.pdf", dpi=600, bbox_inches="tight")  # 也可以保存为 PDF 矢量图
plt.close()
