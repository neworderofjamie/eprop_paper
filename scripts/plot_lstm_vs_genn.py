from pandas import read_csv
import matplotlib.pyplot as plt
import plot_settings

# Load data
data= read_csv("lstm_vs_genn.csv", delimiter=",")


titan_v_data=data[data["Device"] == "Titan V"]

lstm = titan_v_data[(titan_v_data["Model"] == "TF LSTM") & (titan_v_data["Batch size"] > 1)]
genn = titan_v_data[titan_v_data["Model"] == "GeNN LSNN (256 ALIF)"]

fig, axes = plt.subplots(2, sharex=True)
lstm_actor = axes[0].plot(lstm["Batch size"], lstm["Latency per batch [ms]"])[0]
genn_actor = axes[0].plot(genn["Batch size"], genn["Latency per batch [ms]"])[0]
axes[1].plot(lstm["Batch size"], lstm["Energy Delay Product (uJs)"])
axes[1].plot(genn["Batch size"], genn["Energy Delay Product (uJs)"])


axes[1].set_xlabel("Batch size")
axes[0].set_ylabel("Latency per batch [ms]")
axes[1].set_ylabel("Energy Delay Product [uJs]")

fig.legend([lstm_actor, genn_actor], ["LSTM (TF)", "LSNN (GeNN)"], 
           loc="lower center", ncol=2)
fig.tight_layout(pad=0, h_pad=1.0, rect=[0.0, 0.075, 1.0, 1.0])
plt.show()