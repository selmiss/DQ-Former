import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager


prop = font_manager.FontProperties(fname="./results/Times New Roman.ttf", size=36)
prop_bold = font_manager.FontProperties(
    fname="./results/Times New Roman - Bold.ttf", size=30
)
# Load data into pandas dataframe

lith_dict = {
    "LITH": [
        "Leave Blank",
        "Sandstone",
        "Sandstone/Shale",
        "Chalk",
        "Limestone",
    ],
    "COUNT": [0, 57.81, 67.08, 66.50, 74.55],
    "TASK": "Structure"
}

lith_dict2 = {
    "LITH": [
        "Leave Blank",
        "Sandstone",
        "Sandstone/Shale",
        "Chalk",
        "Limestone",
    ],
    "COUNT": [0, 47.06, 66.77, 64.77, 72.39],
    "TASK": "Source"
}

lith_dict3 = {
    "LITH": [
        "Leave Blank",
        "Sandstone",
        "Sandstone/Shale",
        "Chalk",
        "Limestone",
    ],
    "COUNT": [0, 32.44, 41.06, 42.17, 50.71],
    "TASK": "Property"
}

lith_dict4 = {
    "LITH": [
        "",
        "DyQ-Former",
        "w/o Entropy Pathcing",
        "w/o Dynamic Query",
        "LLM Only",
    ],
    "COUNT": [0, 40.78, 40.83, 42.02, 48.58],
    "TASK": "Application"
}

number_offset = [1.6, 0.75, 0.28, 0.29, 0.1]

name = lith_dict4["TASK"]
df = pd.DataFrame.from_dict(lith_dict4)
# df = pd.read_csv("data.csv")   # 默认第一行是列名
# print(df)
# Get key properties for colours and labels
# max_value_full_ring = max(df["COUNT"])
max_value_full_ring = 60

ring_colours = [
    # "#2f4b7c",
    # "#665191",
    "#a05195",
    "#d45087",
    "#f95d6a",
    "#ff7c43",
    "#ffa600",
]

ring_labels = [f"   {x} ({v}) " for x, v in zip(list(df["LITH"]), list(df["COUNT"]))]
data_len = len(df)

# Begin creating the figure
fig = plt.figure(figsize=(10, 10), linewidth=10, edgecolor="white", facecolor="white")

rect = [0.1, 0.1, 0.8, 0.8]

# Add axis for radial backgrounds
PCT_MAX = max_value_full_ring


def pct2rad(p):  # p 可以是标量或 numpy 数组
    return (np.asarray(p) / PCT_MAX) * 2 * np.pi


ax_polar_bg = fig.add_axes(rect, polar=True, frameon=False)
ax_polar_bg.set_theta_zero_location("N")
ax_polar_bg.set_theta_direction(-1)
ticks_pct = np.array([0, 10, 20, 30, 40, 50])
ax_polar_bg.set_xticks(pct2rad(ticks_pct))
ax_polar_bg.set_xticklabels([str(t) for t in ticks_pct], fontproperties=prop)
ax_polar_bg.yaxis.grid(False)
ax_polar_bg.set_yticklabels([]) 

# Loop through each entry in the dataframe and plot a grey
# ring to create the background for each one
ring_height = 0.8
r = ring_height / 2.0
s = np.pi * (r**2) * 3500  # 自动面积
angle_max = 2

for i in range(data_len):
    if i == 0:
        ax_polar_bg.barh(
            i,
            max_value_full_ring * angle_max * np.pi / max_value_full_ring,
            color="none",
            height=ring_height,
        )
        continue

    ax_polar_bg.barh(
        i,
        max_value_full_ring * angle_max * np.pi / max_value_full_ring,
        color="grey",
        alpha=0.1,
        height=ring_height,
    )
    ax_polar_bg.scatter(0, i, s=s, color=ring_colours[i], zorder=3)

# Hide all axis items
# ax_polar_bg.axis("off")


# Add axis for radial chart for each entry in the dataframe
ax_polar = fig.add_axes(rect, polar=True, frameon=False)
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.xaxis.grid(False)
# ax_polar.set_rgrids([0, 1, 2, 3, 4],
#                     # labels=ring_labels,
#                     angle=0,
#                     fontweight='bold',
#                     color='black', verticalalignment='center', fontproperties=prop_bold)


# Loop through each entry in the dataframe and create a coloured
# ring for each entry
for i in range(data_len):
    val = df["COUNT"][i] * angle_max * np.pi / max_value_full_ring
    if i == 0:
        ax_polar.barh(i, val, color="none", height=ring_height)
        continue
    ax_polar.barh(i, val, color=ring_colours[i], height=ring_height, label=df["LITH"][i])

    # 在末端加一个小圆
    ax_polar.scatter(val, i, s=s, color=ring_colours[i], zorder=3)
    
    ax_polar.text(
        val + number_offset[i],  # θ角度稍微往外偏移，避免和圆点重叠
        i + 0.5,  # 半径位置就是第 i 个环
        f"{df['COUNT'][i]}",
        ha="left",
        va="center",
        color="black",
        fontweight="bold",
        fontproperties=prop_bold,
    )

ax_polar.text(
    0,
    -0.6,
    name,
    ha="center",
    va="center",
    color="black",
    fontweight="bold",
    fontproperties=prop_bold,
    bbox=dict(
        boxstyle="round,pad=0.1,rounding_size=0.5",  # 圆角 + 内边距
        facecolor="lightgreen",  # 背景色
        edgecolor="lightgreen",  # 边框颜色
        linewidth=0.1,  # 边框粗细
        alpha=0.5,  # 透明度（可选）
    ),
)

# Hide all grid elements for the
ax_polar.grid(False)
ax_polar.tick_params(
    axis="both", left=False, bottom=False, labelbottom=False, labelleft=False
)
ax_polar.set_yticks([])
ax_polar.legend(
    loc="upper center",         # 位置
    bbox_to_anchor=(1.2, 1.1), # 偏移
    prop=prop_bold,
    frameon=False              # 是否显示边框
)

plt.savefig(f"./utils/figures/results/{str(name).lower()}.pdf", dpi=600, bbox_inches="tight")
plt.savefig(f"./utils/figures/results/{str(name).lower()}.png", dpi=600, bbox_inches="tight")
plt.close()
