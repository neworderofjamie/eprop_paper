import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plot_settings
import seaborn as sns 
from pandas import read_csv

from six import iterkeys, itervalues

BAR_WIDTH = 1.0
BAR_PAD = 0.2
GROUP_PAD = 1.0

def plot(data, axis, column_name):
    # Extract PyTorch and GeNN data
    pytorch_data = data[data["Learning rule"] == "BPTT"]
    eprop_data = data[data["Learning rule"] == "EProp"]
    event_prop_data = data[data["Learning rule"] == "EventProp"]

    assert np.all(pytorch_data["Num LIF"].values == eprop_data["Num LIF"].values)
    assert np.all(pytorch_data["Num LIF"].values == event_prop_data["Num LIF"].values)

    # Configure bars
    group_size = 3
    num_groups = len(pytorch_data["Num LIF"].values)
    num_bars = group_size * num_groups
    bar_x = np.empty(num_bars)

    # Loop through each group (device) of bars
    group_x = []
    start = 0.0
    for d in range(0, num_bars, group_size):
        end = start + ((BAR_WIDTH + BAR_PAD) * group_size)
        bar_x[d:d + group_size] = np.arange(start, end - 0.1, BAR_WIDTH + BAR_PAD)
        
        group_x.append(start + ((end - BAR_WIDTH - start) * 0.5))

        # Update start for next group
        start = end + GROUP_PAD

    pal = sns.color_palette()
    legend_actors = []
    for i, d in enumerate([pytorch_data, eprop_data, event_prop_data]):
        bars = axis.bar(bar_x[i::group_size], d[column_name].values, 
                        BAR_WIDTH, color=pal[i], linewidth=0)
        legend_actors.append(bars[0])

    # Configure axis
    axis.set_xticks(group_x)
    axis.set_xticklabels([f"{g} neurons" for g in pytorch_data["Num LIF"].values], ha="center")

    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    
    return legend_actors

# Load data
data = read_csv("eprop_event_prop.csv", delimiter=",")

fig, axes = plt.subplots(1, 2, figsize=(plot_settings.double_column_width, 3.0))


# Create evaluation performance panel
plot(data, axes[0], "Testing performance 1 [%]")
axes[0].set_ylabel("Accuracy [%]")
axes[0].set_ylim((90, 100))


# Create training time panel
legend_actors = plot(data, axes[1], "Training time [s]")
axes[1].set_ylabel("Training time [s]")


# Show figure legend with devices beneath figure
fig.legend(legend_actors, ["PyTorch (BPTT LIF FF)", "GeNN (eProp LIF FF)", "GeNN (EventProp LIF FF)"], 
            ncol=3, frameon=False, loc="lower center")

fig.tight_layout(pad=0, rect=[0.0, 0.0 if plot_settings.presentation else 0.1, 1.0, 1.0])


if not plot_settings.presentation:
    fig.savefig("../figures/eprop_event_prop.pdf")
plt.show()
