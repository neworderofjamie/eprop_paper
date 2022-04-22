import matplotlib.pyplot as plt
import numpy as np
from os import path
import plot_settings
import seaborn as sns 

density = np.asarray([1.0, 0.1, 0.05, 0.01])
train_time = np.asarray([5072069.393037, 1266687.215274, 926892.090231, 580000.520527])
train_time_deep_r = np.asarray([None, 1365169.228164, 1010249.199616, 603524.890482])
test_performance = np.asarray([79.28, 77.61, 76.06, 72.08])
test_performance_deep_r = np.asarray([None, 73.100707])

# Plot training time bars
time_bar_width = 0.4
time_tick_x = [0, 1, 2, 3]
time_bar_x = [0, time_tick_x[1] - 0.22, time_tick_x[2] - 0.22, time_tick_x[3] - 0.22]
time_deep_r_bar_x = [time_tick_x[1] + 0.22, time_tick_x[2] + 0.22, time_tick_x[3] + 0.22]

#fig, axes = plt.subplots(1, 2, figsize=(898 / 96.0, 500 / 96.0), dpi=96)
dpi = 200
fig, axis = plt.subplots(figsize=(898.0 / dpi, 500.0 / dpi), dpi=dpi)

time_axis = axis
time_actor = time_axis.bar(time_bar_x, train_time / 60000, width=time_bar_width)
deep_r_time_actor = time_axis.bar(time_deep_r_bar_x, train_time_deep_r[1:] / 60000, width=time_bar_width)

time_axis.set_xticks(time_tick_x)
time_axis.set_xticklabels(density)
time_axis.set_xlabel("Connection density")
time_axis.set_ylabel("Training time [minutes]")
#time_axis.set_title("A", loc="left")

# Remove axis junk
sns.despine(ax=time_axis)
time_axis.xaxis.grid(False)

"""
axes[1].set_xlabel("Connection density")
axes[1].set_ylabel("Accuracy [%]")
axes[1].set_title("B", loc="left")

# Remove axis junk
sns.despine(ax=axes[1])
axes[1].xaxis.grid(False)
"""
# Add figure legend
fig.legend([time_actor, deep_r_time_actor], ["Fixed connectivity", "Deep-R rewiring"],
           ncol=2, loc="lower center")
fig.tight_layout(pad=0.0, rect=[0.005, 0.1, 1.0, 1.0])
fig.savefig("../figures/shd_deep_r.png")
plt.show()
