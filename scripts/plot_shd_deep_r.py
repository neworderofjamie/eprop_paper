import matplotlib.pyplot as plt
import numpy as np
from os import path
import plot_settings
import seaborn as sns 

density = [1.0, 0.1, 0.05, 0.01]
train_time = [6375, 4970, 4850, 4775]
train_time_deep_r = [None, 4990, 4930, 4780]
test_performance = [79.063604, 73.498233]
test_performance_deep_r = [None, 73.100707]

# Plot training time bars
time_bar_width = 0.4
time_tick_x = [0, 1, 2, 3]
time_bar_x = [0, time_tick_x[1] - 0.22, time_tick_x[2] - 0.22, time_tick_x[3] - 0.22]
time_deep_r_bar_x = [time_tick_x[1] + 0.22, time_tick_x[2] + 0.22, time_tick_x[3] + 0.22]

fig, axes = plt.subplots(1, 2, figsize=(898 / 96.0, 500 / 96.0), dpi=96)

time_actor = axes[0].bar(time_bar_x, train_time, width=time_bar_width)
deep_r_time_actor = axes[0].bar(time_deep_r_bar_x, train_time_deep_r[1:], width=time_bar_width)

axes[0].set_xticks(time_tick_x)
axes[0].set_xticklabels(density)
axes[0].set_xlabel("Connection density")
axes[0].set_ylabel("Batch time [ms]")
axes[0].set_title("A", loc="left")

axes[0].set_xlabel("Connection density")
axes[0].set_ylabel("Accuracy [%]")
axes[1].set_title("B", loc="left")

# Remove axis junk
sns.despine(ax=axes[0])
sns.despine(ax=axes[1])
axes[0].xaxis.grid(False)
axes[1].xaxis.grid(False)

# Add figure legend
fig.legend([time_actor, deep_r_time_actor], ["Fixed connectivity", "Deep-R rewiring"],
           ncol=2, loc="lower center")
fig.tight_layout(pad=0, rect=[0.0, 0.075, 1.0, 1.0])
fig.savefig("../figures/shd_deep_r.png")
plt.show()