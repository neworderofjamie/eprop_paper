import matplotlib.pyplot as plt
import numpy as np
from os import path
import plot_settings
import seaborn as sns 

density = np.asarray([1.0, 0.1, 0.05, 0.01])
train_time = np.asarray([5072069.393037, 1266687.215274, 926892.090231, 580000.520527])
train_time_deep_r = np.asarray([None, 1365169.228164, 1010249.199616, 603524.890482])
test_performance = np.asarray([74.690813, 73.674912, 70.538869, 62.279152])
test_performance_deep_r = np.asarray([None, 72.924028, 69.169611, 62.279152])

# Plot training time bars
time_bar_width = 0.4
time_tick_x = [0, 1, 2, 3]
time_bar_x = [0, time_tick_x[1] - 0.22, time_tick_x[2] - 0.22, time_tick_x[3] - 0.22]
time_deep_r_bar_x = [time_tick_x[1] + 0.22, time_tick_x[2] + 0.22, time_tick_x[3] + 0.22]

dpi = 200
fig, axes = plt.subplots(1, 2, figsize=(898 / dpi, 500 / dpi), dpi=dpi)

time_axis = axes[0]
time_actor = time_axis.bar(time_bar_x, train_time / 60000, width=time_bar_width)
deep_r_time_actor = time_axis.bar(time_deep_r_bar_x, train_time_deep_r[1:] / 60000, width=time_bar_width)

time_axis.set_xticks(time_tick_x)
time_axis.set_xticklabels(density)
time_axis.set_xlabel("Connection density")
time_axis.set_ylabel("Training time [minutes]")
time_axis.set_title("A", loc="left")

# Remove axis junk
sns.despine(ax=time_axis)
time_axis.xaxis.grid(False)

perf_axis = axes[1]
perf_axis.bar(time_bar_x, test_performance, width=time_bar_width)
perf_axis.bar(time_deep_r_bar_x, test_performance_deep_r[1:], width=time_bar_width)

perf_axis.set_xticks(time_tick_x)
perf_axis.set_xticklabels(density)
perf_axis.set_xlabel("Connection density")
perf_axis.set_ylabel("Accuracy [%]")
perf_axis.set_title("B", loc="left")

# Remove axis junk
sns.despine(ax=perf_axis)
perf_axis.xaxis.grid(False)

# Add figure legend
fig.legend([time_actor, deep_r_time_actor], ["Fixed connectivity", "Deep-R rewiring"],
           ncol=2, loc="lower center")
fig.tight_layout(pad=0.0, rect=[0.005, 0.1, 1.0, 1.0])
fig.savefig("../figures/shd_deep_r.png")
plt.show()
