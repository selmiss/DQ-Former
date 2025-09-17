import matplotlib.pyplot as plt
import numpy as np

# 指标
labels = ["Strct.", "Src.", "Prop.", "App.", "Avg.", "Total"]
num_vars = len(labels)

# 数据
baseline = np.ones(num_vars) * 100
model = np.array([77.87, 75.79, 50.2, 48.58, 82.31, 75])

# 角度
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection':'polar'})

# baseline 外圈
ax.bar(angles, baseline, width=2*np.pi/num_vars, 
       color="lightgray", alpha=0.3, edgecolor="black", label="Baseline (100)")

# 模型 内圈
ax.bar(angles, model, width=2*np.pi/num_vars, 
       color="royalblue", alpha=0.7, edgecolor="black", label="Model")

# 设置角度刻度
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=12)

# 去掉径向标签，只留圆环
ax.set_yticklabels([])

ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.savefig("./results/round_bar.png", dpi=600, bbox_inches="tight")
plt.close()
