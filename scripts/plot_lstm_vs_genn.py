from pandas import read_csv
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns

# Load data
data= read_csv("lstm_vs_genn.csv", delimiter=",")

# Extract the data for different platforms
loihi_data=data[data["Device"] == "Loihi"]
titan_v_data=data[data["Device"] == "Titan V"]

assert len(loihi_data) == 1

lstm_data = titan_v_data[(titan_v_data["Model"] == "TF LSTM") & (titan_v_data["Batch size"] > 1)]
genn_data = titan_v_data[titan_v_data["Model"] == "GeNN LSNN (256 ALIF)"]

fig, axes = plt.subplots(2, sharex=True)

pal = sns.color_palette()
lstm_actor = axes[0].plot(lstm_data["Batch size"], lstm_data["Latency per batch [ms]"], color=pal[0])[0]
genn_actor = axes[0].plot(genn_data["Batch size"], genn_data["Latency per batch [ms]"], color=pal[1])[0]
loihi_actor = axes[0].scatter([1], loihi_data["Latency per batch [ms]"], color=pal[2], marker="X")
axes[1].plot(lstm_data["Batch size"], lstm_data["Energy Delay Product (uJs)"], color=pal[0])
axes[1].plot(genn_data["Batch size"], genn_data["Energy Delay Product (uJs)"], color=pal[1])
axes[1].scatter([1], loihi_data["Energy Delay Product (uJs)"], color=pal[2], marker="X")

axes[1].set_xlabel("Batch size")
axes[0].set_ylabel("Batch time [ms]")
axes[1].set_ylabel("EDP [uJs]")

fig.align_ylabels([axes[0], axes[1]])
fig.legend([lstm_actor, genn_actor, loihi_actor], ["LSTM (TF)", "LSNN (GeNN)", "LSSN (Loihi)"], 
           loc="lower center", ncol=3)
fig.tight_layout(pad=0, h_pad=1.0, rect=[0.0, 0.075, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/lstm_vs_genn.eps")
plt.show()
