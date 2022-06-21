import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plot_settings
import seaborn as sns 

from collections import namedtuple
from glob import glob
from itertools import product
from six import iterkeys, itervalues

BAR_WIDTH = 1.0
BAR_PAD = 0.2
GROUP_PAD = 1.0

Performance = namedtuple("Performance", ["name", "test_mean", "test_sd"])

class Config(object):
    def __init__(self, name, experiment_files, references):
        self.name = name
        self.performances = references

        # Assert that there are some experiment files
        assert len(experiment_files) > 0
        
        # Loop through experiment files
        for name, file_prefix in experiment_files:
            # Find all directories containing runs of this experiment
            data_directories = glob(os.path.join("performance_data", file_prefix))# + "_100_epochs_*"))
            
            # Read test performance
            test_performance = [self._get_test_performance(d) for d in data_directories]

            ## Add performance tuple
            self.performances.append(Performance(name, np.mean(test_performance), np.std(test_performance)))
    
    def _get_test_performance(self, path):
        # Find evaluation files, sorting numerically
        evaluate_files = list(sorted(glob(os.path.join(path, "performance_evaluate_*.csv")),
                                    key=lambda x: int(os.path.basename(x)[21:-4])))
        
        assert len(evaluate_files) > 0
        
        # Load last evaluate file
        test_data = np.loadtxt(evaluate_files[-1], delimiter=",", skiprows=1)

        # Calculate performance
        num_trials = np.sum(test_data[:,1])
        num_correct = np.sum(test_data[:,2])

        # Add to list
        return 100.0 * (num_correct / num_trials)


def plot(configs):
    pal = sns.color_palette()
    colour_map = {}

    bar_x = []
    bar_height = []
    bar_colour = []
    bar_error = []
    group_x = []

    tick_label = []

    # Loop through configurations
    last_x = 0.0
    for c in configs:
        # Calculate centre of this group
        group_centre = ((len(c.performances) - 1) / 2.0) * (BAR_PAD + BAR_WIDTH)
        group_x.append(last_x + group_centre)
        
        # Loop through all performances achieved in this config
        for p in c.performances:
            # Assign this performance record a colour based on its name
            if p.name in colour_map:
                colour = colour_map[p.name]
            else:
                colour = pal[len(colour_map)]
                colour_map[p.name] = colour

            # Add testing bar
            bar_x.append(last_x)
            bar_height.append(p.test_mean)
            bar_colour.append(colour)
            bar_error.append(p.test_sd)
            
            # Use name for tick
            tick_label.append(p.name)
            
            # Add spacing between bars
            last_x += (BAR_WIDTH + BAR_PAD)
            
        # Add extra padding between groups
        last_x += (GROUP_PAD - BAR_PAD)

    fig, axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))
    actors = axis.bar(bar_x, bar_height, BAR_WIDTH, yerr=bar_error, color=bar_colour)

    axis.set_xticks(group_x)
    axis.set_xticklabels([c.name for c in configs], ha="center")
    axis.set_ylabel("Accuracy [%]")
    axis.set_ylim((0, 100.0))

    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)

    fig.legend([mpatches.Rectangle(color=c, width=10, height=10, xy=(0,0)) for c in itervalues(colour_map)], 
               iterkeys(colour_map), loc="lower center", ncol=len(colour_map) if plot_settings.presentation else 2)
    fig.tight_layout(pad=0, rect=[0.0, 0.025 if plot_settings.presentation else 0.2, 1.0, 1.0])
    return fig

shd_configs = [Config("256 neurons", 
                     [("GeNN (eProp LSNN FF)", "shd_256_feedforward_100_epochs_*"), ("GeNN (eProp LSNN RC)", "shd_256_100_epochs_*")],
                     [Performance("PyTorch (BPTT LIF FF)", 74.0, 1.7), Performance("PyTorch (BPTT LIF RC)", 80.0, 2.0)]),
              Config("512 neurons", 
                     [("GeNN (eProp LSNN FF)", "shd_512_feedforward_100_epochs_*"), ("GeNN (eProp LSNN RC)", "shd_512_100_epochs_*")],
                     []),
              Config("1024 neurons", 
                     [("GeNN (eProp LSNN FF)", "shd_1024_feedforward_100_epochs_*"), ("GeNN (eProp LSNN RC)", "shd_1024_100_epochs_*")],
                     [])]

smnist_configs = [Config("256 neurons", 
                         [("GeNN (eProp LSNN RC)", "smnist_256_100_epochs_*")],
                         [Performance("TensorFlow (BPTT LSNN RC)", 96.4, 0.0), Performance("TensorFlow (BPTT LIF RC)", 63.3, 0.0)]),
                  Config("512 neurons", 
                         [("GeNN (eProp LSNN RC)", "smnist_512_100_epochs_*")],
                         []),
                  Config("1024 neurons", 
                         [("GeNN (eProp LSNN RC)", "smnist_1024_100_epochs_*")],
                         [])]

connectivities = [0.01, 0.05, 0.1]
experiments = [("Fixed", ""), ("Deep-R 80:20", "_deep_r_80_20"), ("Deep-R 50:50", "_deep_r_50_50")]
sparse_configs = [Config(f"{in_rec * 100}% input-recurrent\n{rec_rec * 100}% recurrent-recurrent",
                         [(e[0], f"shd_256_new_sparse_{in_rec}_{rec_rec}{e[1]}") for e in experiments],
                         [])
                  for in_rec, rec_rec in product(connectivities, repeat=2)]
                                 
shd_fig = plot(shd_configs)
smnist_fig = plot(smnist_configs)
sparse_fig = plot(sparse_configs)                         

if not plot_settings.presentation:
    shd_fig.savefig("../figures/shd_performance.pdf")
    smnist_fig.savefig("../figures/smnist_performance.pdf")
    sparse_fig.savefig("../figures/sparse_performance.pdf")

plt.show()
