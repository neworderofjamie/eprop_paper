from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plot_settings
import seaborn as sns

# Load data
data = read_csv("multi_gpu.csv", delimiter=",")

# Get reference time
reference_time = data[(data["Device"] == "V100") & (data["Num GPUs"] == 1)]["Training time [ms]"].values[0]

fig, axis = plt.subplots()

# Loop through unique devices
device_actors = []
devices = data["Device"].unique()
for d in devices:
    # Get device data
    device_data = data[data["Device"] == d]

    # Plot speedup compared to reference time
    speedup = reference_time / device_data["Training time [ms]"]
    actor = axis.plot(device_data["Num GPUs"], speedup,
                      marker="x")[0]
    
    # In dashed line show perfect scaling
    axis.plot(device_data["Num GPUs"], speedup.values[0] * device_data["Num GPUs"],
              linestyle="--", color=actor.get_color())
    device_actors.append(actor)

axis.set_xscale("log", basex=2)
axis.set_yscale("log", basey=2)
axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
axis.set_xlabel("Num GPUs")
axis.set_ylabel("Speedup compared to single V100")

# Remove axis junk
sns.despine(ax=axis)

fig.legend(device_actors, devices, loc="lower center", ncol=2)
fig.tight_layout(pad=0, rect=[0.0, 0.0 if plot_settings.presentation else 0.075, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/multi_gpu.eps")
plt.show()
