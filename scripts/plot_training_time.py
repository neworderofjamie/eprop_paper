import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plot_settings
import seaborn as sns

# Load data
data = read_csv("training_time.csv", delimiter=",")

# Count unique neuron counts
num_neurons = data["Num hidden neurons"].unique()
timesteps = data["Timesteps"].unique()

# Create an axis for each model size
fig, axes = plt.subplots(1, len(num_neurons), sharey=True)

# Extract PyTorch and GeNN data
pytorch_data = data[data["Algorithm"] == "BPTT (PyTorch)"]
genn_data = data[data["Algorithm"] == "eProp (GeNN)"]

pytorch_data = pytorch_data.pivot(index="Num hidden neurons", columns="Timesteps", values="Batch time [ms]")
genn_data = genn_data.pivot(index="Num hidden neurons", columns="Timesteps", values="Batch time [ms]")

assert np.all(pytorch_data.index.values == genn_data.index.values)

tick_x = np.arange(0.2, len(pytorch_data.index.values), 1.0)
genn_x = np.arange(0.0, len(pytorch_data.index.values), 1.0)
pytorch_x = np.arange(0.4, len(pytorch_data.index.values), 1.0)

# Loop through data grouped by number of hidden neurons
for i, (ax, n) in enumerate(zip(axes, pytorch_data.index)):
    # Plot bars
    genn_actor = ax.bar(genn_x, genn_data.loc[n].values, 0.4)
    pytorch_actor = ax.bar(pytorch_x, pytorch_data.loc[n].values, 0.4)
    
    # Configure ticks
    ax.set_xticks(tick_x)
    ax.set_xticklabels(genn_data.loc[n].index.values)
    
    # Configure axis
    ax.set_title("%u neurons" % n)
    ax.set_xlabel("Timesteps")
    
    # Remove axis junk
    sns.despine(ax=ax, left=(i != 0))
    ax.xaxis.grid(False)

axes[0].set_ylabel("Batch time [ms]")
fig.legend([genn_actor, pytorch_actor], ["GeNN (eProp)", "PyTorch (BPTT)"], loc="lower center", ncol=2)
fig.tight_layout(pad=0, rect=[0.0, 0.0 if plot_settings.presentation else 0.075, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/training_time.eps")
plt.show()
