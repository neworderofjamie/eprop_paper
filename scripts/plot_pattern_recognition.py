import matplotlib.pyplot as plt
import numpy as np
from os import path
import plot_settings
import seaborn as sns 

from glob import glob
from itertools import chain

def get_data_lists(data, sparsity, column):
    sparse_sparse_average = [np.average(data[(s, s)][column]) for s in sparsity]
    sparse_sparse_std = [np.std(data[(s, s)][column]) for s in sparsity]
    
    dense_sparse_average = [np.average(data[(1.0, s)][column]) for s in sparsity]
    dense_sparse_std = [np.std(data[(1.0, s)][column]) for s in sparsity]
    
    return ((sparse_sparse_average, sparse_sparse_std),
            (dense_sparse_average, dense_sparse_std))

def get_interleaved_data_lists(data, sparsity, column):
    sparse_sparse, dense_sparse = get_data_lists(data, sparsity, column)
    
    return (list(chain(*zip(sparse_sparse[0], dense_sparse[0]))),
            list(chain(*zip(sparse_sparse[1], dense_sparse[1]))))

output_filenames = list(glob("pattern_recognition/output*.txt"))
data = {}
for f in output_filenames:
    title = path.splitext(path.split(f)[1])[0]
    title_components = title.split("_")
    input_recurrent_sparsity = float(title_components[1])
    recurrent_recurrent_sparsity = float(title_components[2])

    neuron_update_times = []
    presynaptic_update_times = []
    synapse_dynamics_times = []
    mse = []
    with open(f) as file:
        for line in file:
            if line.startswith("Neuron update:"):
                neuron_update_times.append(float(line[15:]))
            elif line.startswith("Presynaptic update:"):
                presynaptic_update_times.append(float(line[20:]))
            elif line.startswith("Synapse dynamics:"):
                synapse_dynamics_times.append(float(line[18:]))
            elif line.startswith("9: Total MSE:"):
                mse.append(float(line[14:]))

    data[(input_recurrent_sparsity, recurrent_recurrent_sparsity)] =\
        (neuron_update_times, presynaptic_update_times, synapse_dynamics_times, mse)


sparsity = sorted(list(set(r for i, r in data.keys())))
pal = sns.color_palette()
 
#---------------------------------------
# Plot MSE with sparsity
#---------------------------------------
mse_fig, mse_axis = plt.subplots()

sparse_sparse_mse, dense_sparse_mse = get_data_lists(data, sparsity, 3)

tick_x = np.arange(len(sparsity))
sparse_sparse_x = tick_x - 0.22
dense_sparse_x = tick_x + 0.22

sparse_sparse = mse_axis.bar(sparse_sparse_x, sparse_sparse_mse[0], yerr=sparse_sparse_mse[1],
                             width=0.4, color=pal[0])
dense_sparse = mse_axis.bar(dense_sparse_x, dense_sparse_mse[0], yerr=dense_sparse_mse[1], 
                            width=0.4, color=pal[1])

mse_axis.set_ylabel("Output MSE")
mse_axis.set_xlabel("Sparsity")
mse_axis.set_xticks(tick_x)
mse_axis.set_xticklabels(sparsity)

# Remove axis junk
sns.despine(ax=mse_axis)
mse_axis.xaxis.grid(False)

mse_fig.legend([sparse_sparse, dense_sparse], ["Sparse, sparse", "Dense, sparse"],
               loc="lower center", ncol=2)
mse_fig.tight_layout(pad=0, rect=[0.0, 0.0 if plot_settings.presentation else 0.075, 1.0, 1.0])

#---------------------------------------
# Plot performance with sparsity
#---------------------------------------
perf_fig, perf_axis = plt.subplots()

bar_x = np.empty(len(sparse_sparse_x) + len(dense_sparse_x))
bar_x[0::2] = sparse_sparse_x
bar_x[1::2] = dense_sparse_x

neuron_update_mean, neuron_update_std = get_interleaved_data_lists(data, sparsity, 0)
presynaptic_update_mean, presynaptic_update_std = get_interleaved_data_lists(data, sparsity, 1)
synapse_dynamics_mean, synapse_dynamics_std = get_interleaved_data_lists(data, sparsity, 2)

perf_axis.bar(bar_x, neuron_update_mean, width=0.4)
perf_axis.bar(bar_x, presynaptic_update_mean, width=0.4, bottom=neuron_update_mean)
perf_axis.bar(bar_x, synapse_dynamics_mean, width=0.4, 
              bottom=np.add(neuron_update_mean, presynaptic_update_mean))

perf_axis.set_ylabel("Time [s]")
perf_axis.set_xlabel("Sparsity")
perf_axis.set_xticks(tick_x)
perf_axis.set_xticklabels(sparsity)

# Remove axis junk
sns.despine(ax=perf_axis)
perf_axis.xaxis.grid(False)

#perf_fig.legend([sparse_sparse, dense_sparse], ["Sparse, sparse", "Dense, sparse"],
#                loc="lower center", ncol=2)
perf_fig.tight_layout(pad=0)#, rect=[0.0, 0.0 if plot_settings.presentation else 0.075, 1.0, 1.0])


plt.show()