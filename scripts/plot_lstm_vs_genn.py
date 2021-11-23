from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
import plot_settings
import seaborn as sns

def remove_axis_junk(axis):
    sns.despine(ax=axis)
    axis.xaxis.grid(False)

# Load data
data = read_csv("lstm_vs_genn.csv", delimiter=",")

# Extract the data for different platforms
loihi_data = data[data["Device"] == "Loihi"]
genn_cpu_data = data[data["Device"] == "Intel Core i5-9400F"]
tf_cpu_data = data[data["Device"] == "Intel Core i5-7440HQ"]
titan_v_data = data[data["Device"] == "Titan V"]

assert len(loihi_data) == 1
assert len(genn_cpu_data) == 1
assert len(tf_cpu_data) == 1

# Split Titan V data up into LSTM and GeNN and batch 1 and batch N
lstm_batch_n_data = titan_v_data[(titan_v_data["Model"] == "TF LSTM") & (titan_v_data["Batch size"] > 1)]
genn_batch_n_data = titan_v_data[(titan_v_data["Model"] == "GeNN LSNN (256 ALIF)") & (titan_v_data["Batch size"] > 1)]

lstm_batch_1_data = titan_v_data[(titan_v_data["Model"] == "TF LSTM") & (titan_v_data["Batch size"] == 1)]
genn_batch_1_data = titan_v_data[(titan_v_data["Model"] == "GeNN LSNN (256 ALIF)") & (titan_v_data["Batch size"] == 1)]

assert len(lstm_batch_1_data) == 1
assert len(genn_batch_1_data) == 1

fig = plt.figure(figsize=(plot_settings.double_column_width, 2.5))


# Create outer gridspec to divide figure into 4 columns
gsp = gs.GridSpec(1, 4)

# Divide first column and remaining 3 columns into two rows
batch_1_gs = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[0])
batch_n_gs =  gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[1:])

# Create axes for each gridspec
batch_1_latency_axis = plt.Subplot(fig, batch_1_gs[0])
batch_1_edp_axis = plt.Subplot(fig, batch_1_gs[1], sharex=batch_1_latency_axis)
batch_n_latency_axis = plt.Subplot(fig, batch_n_gs[0], sharey=batch_1_latency_axis)
batch_n_edp_axis = plt.Subplot(fig, batch_n_gs[1], sharex=batch_n_latency_axis, sharey=batch_1_edp_axis)

# Add axes to figure
fig.add_subplot(batch_1_latency_axis)
fig.add_subplot(batch_1_edp_axis)
fig.add_subplot(batch_n_latency_axis)
fig.add_subplot(batch_n_edp_axis)

# Plot lines for GeNN vs TF latency and EDP
pal = sns.color_palette()
lstm_gpu_actor = batch_n_latency_axis.plot(lstm_batch_n_data["Batch size"], 
                                           lstm_batch_n_data["Latency per batch [ms]"], 
                                           color=pal[1], marker="x")[0]
genn_gpu_actor = batch_n_latency_axis.plot(genn_batch_n_data["Batch size"], 
                                           genn_batch_n_data["Latency per batch [ms]"], 
                                           color=pal[2], marker="x")[0]

batch_n_edp_axis.plot(lstm_batch_n_data["Batch size"], 
                      lstm_batch_n_data["Energy Delay Product (uJs)"], 
                      color=pal[1], marker="x")[0]
batch_n_edp_axis.plot(genn_batch_n_data["Batch size"], 
                      genn_batch_n_data["Energy Delay Product (uJs)"], 
                      color=pal[2], marker="x")[0]

# Plot bars for various batch size 1 things
bar_x = list(range(5))

bar_data_frames = [(tf_cpu_data, "TensorFlow CPU\n(LSTM RC)*"),
                   (lstm_batch_1_data, "TensorFlow GPU\n(LSTM RC)"), 
                   (genn_batch_1_data, "GeNN GPU\n(LSNN RC)"),
                   (genn_cpu_data, "GeNN CPU\n(LSNN RC)"),
                   (loihi_data, "Loihi\n(LSNN RC)*")]
latency_bar_height = [f["Latency per batch [ms]"].iloc[0] for f,_ in bar_data_frames]
edp_bar_height = [f["Energy Delay Product (uJs)"].iloc[0] for f,_ in bar_data_frames]
bar_colours = [pal[i] for i,_ in enumerate(bar_data_frames)]

batch_1_latency_axis.bar(bar_x, latency_bar_height, 0.8, color=bar_colours)
batch_1_edp_axis.bar(bar_x, edp_bar_height, 0.8, color=bar_colours)

batch_1_edp_axis.set_xticks(bar_x)
batch_1_edp_axis.set_xticklabels([n for _, n in bar_data_frames], rotation=90, ma="right")


batch_n_edp_axis.set_xlabel("Batch size")
batch_1_latency_axis.set_ylabel("Batch time [ms]")
batch_1_edp_axis.set_ylabel("EDP [uJs]")
batch_n_edp_axis.set_yscale("log")
batch_n_edp_axis.set_ylim((1.0, 1E6))

# Hide unnecessary labels
plt.setp(batch_n_edp_axis.get_yticklabels(), visible=False)
plt.setp(batch_n_latency_axis.get_xticklabels(), visible=False)
plt.setp(batch_n_latency_axis.get_yticklabels(), visible=False)
plt.setp(batch_1_latency_axis.get_xticklabels(), visible=False)

# Remove axis junk
remove_axis_junk(batch_1_latency_axis)
remove_axis_junk(batch_1_edp_axis)
remove_axis_junk(batch_n_latency_axis)
remove_axis_junk(batch_n_edp_axis)
    
fig.align_ylabels([batch_1_latency_axis, batch_1_edp_axis])
fig.legend([lstm_gpu_actor, genn_gpu_actor], 
           ["TensorFlow GPU (LSTM RC)", "GeNN GPU (LSNN RC)"], 
           loc="lower center", ncol=2, bbox_to_anchor=(0.625, 0.325) if plot_settings.presentation else (0.625, 0.1))
fig.tight_layout(pad=0, w_pad=0.5, rect=[0.0, 0.02, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/lstm_vs_genn.pdf")
plt.show()
