from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plot_settings

# Load data
data = read_csv("multi_gpu.csv", delimiter=",")

fig, axis = plt.subplots()

# Loop through unique devices
device_actors = []
devices = data["Device"].unique()
for d in devices:
    # Get device data
    device_data = data[data["Device"] == d]

    # Plot reciprocal of training time
    actor = axis.plot(device_data["Num GPUs"], device_data["Reciprocal training time"],
                      marker="x")[0]

    # In dashed line show perfect scaling
    axis.plot(device_data["Num GPUs"], device_data["Perfect scaling reciprocal training time"],
              linestyle="--", color=actor.get_color())
    device_actors.append(actor)

axis.set_xscale("log", basex=2)
axis.set_yscale("log", basey=2)
axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
axis.set_xlabel("Num GPUs")
axis.set_ylabel("Reciprocol training time")

fig.legend(device_actors, devices, loc="lower center", ncol=2)
fig.tight_layout(pad=0, rect=[0.0, 0.075, 1.0, 1.0])
plt.show()