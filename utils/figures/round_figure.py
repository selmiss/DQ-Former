import matplotlib.pyplot as plt
import numpy as np

# 指标
labels = ["Strct.", "Src.", "Prop.", "App.", "Avg.", "Total"]
num_vars = len(labels)

# baseline (100)
baseline = np.ones(num_vars) * 100

# 模型数据 (示例)
model_scores = np.array([77.87, 75.79, 50.2, 48.58, 82.31, 75])

# 角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
baseline = np.concatenate((baseline, [baseline[0]]))
model_scores = np.concatenate((model_scores, [model_scores[0]]))
angles += angles[:1]

# 画图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# baseline 外圈
ax.plot(angles, baseline, color="lightgray", linewidth=2, linestyle="--", label="Baseline (100)")
ax.fill(angles, baseline, color="lightgray", alpha=0.1)

# 模型数据 内圈
ax.plot(angles, model_scores, color="royalblue", linewidth=2, label="Model")
ax.fill(angles, model_scores, color="royalblue", alpha=0.25)

# 设置坐标轴
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10)
ax.set_ylim(0, 100)

ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.savefig("./results/round_figure.png", dpi=600, bbox_inches="tight")
plt.close()
