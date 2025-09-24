import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
# 指标
labels = ["0.6B", "1.7B", "4B", "8B", "14B"]
num_vars = len(labels)
prop = font_manager.FontProperties(fname="./results/Times New Roman.ttf", size=12)
# 数据
baseline = np.ones(num_vars) * 100
model = np.array([27.23, 26.52, 36.82, 54.79, 56.84])

# 角度
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection':'polar'})

# baseline 外圈
ax.bar(angles, baseline, width=2*np.pi/num_vars, 
       color="lightgray", alpha=0.3, edgecolor="black")

# 模型 内圈
ax.bar(angles, model, width=2*np.pi/num_vars, 
       color="royalblue", alpha=0.7, edgecolor="black")

# 设置角度刻度
ax.set_xticks(angles)
ax.set_yticks([10, 20, 30, 40, 50, 60])
ax.set_ylim(10, 60)
ax.set_xticklabels([])
# ax.set_xticklabels(labels, fontsize=12, fontproperties=prop)

# 去掉径向标签，只留圆环

# ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), prop=prop)
plt.savefig("./utils/figures/results/round_bar.svg", dpi=600, bbox_inches="tight")
plt.close()
