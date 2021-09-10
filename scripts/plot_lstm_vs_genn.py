from pandas import read_csv
import matplotlib.pyplot as plt
import plot_settings

# Load data
data= read_csv("lstm_vs_genn.csv", delimiter=",")

# Extract the data for different platforms
loihi_data=data[data["Device"] == "Loihi"]
titan_v_data=data[data["Device"] == "Titan V"]

assert len(loihi_data) == 1

lstm_data = titan_v_data[(titan_v_data["Model"] == "TF LSTM") & (titan_v_data["Batch size"] > 1)]
genn_data = titan_v_data[titan_v_data["Model"] == "GeNN LSNN (256 ALIF)"]

fig, axes = plt.subplots(2, sharex=True)

lstm_actor = axes[0].plot(lstm_data["Batch size"], lstm_data["Latency per batch [ms]"])[0]
genn_actor = axes[0].plot(genn_data["Batch size"], genn_data["Latency per batch [ms]"])[0]
axes[1].plot(lstm_data["Batch size"], lstm_data["Energy Delay Product (uJs)"])
axes[1].plot(genn_data["Batch size"], genn_data["Energy Delay Product (uJs)"])

axes[0].axhline(loihi_data["Latency per batch [ms]"].iat[0], linestyle="--")
axes[1].axhline(loihi_data["Energy Delay Product (uJs)"].iat[0], linestyle="--")

axes[1].set_xlabel("Batch size")
axes[0].set_ylabel("Latency per batch [ms]")
axes[1].set_ylabel("Energy Delay Product [uJs]")

fig.legend([lstm_actor, genn_actor], ["LSTM (TF)", "LSNN (GeNN)"], 
           loc="lower center", ncol=2)
fig.tight_layout(pad=0, h_pad=1.0, rect=[0.0, 0.075, 1.0, 1.0])
plt.show()