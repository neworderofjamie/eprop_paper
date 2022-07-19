import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plot_settings
import seaborn as sns 

from collections import namedtuple
from glob import glob
from itertools import product
from scipy.stats import wilcoxon
from six import iterkeys, itervalues

BAR_WIDTH = 1.0
BAR_PAD = 0.2
GROUP_PAD = 1.0

Performance = namedtuple("Performance", ["name", "data", "test_mean", "test_sd"])

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
            self.performances.append(Performance(name, test_performance, 
                                                 np.mean(test_performance), 
                                                 np.std(test_performance)))
    
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


def plot_simple(config, width=plot_settings.column_width, height=2.0, x_label=None):
    bar_x = np.arange(len(config.performances)) * (BAR_PAD + BAR_WIDTH)
    bar_height = [p.test_mean for p in config.performances]
    bar_error = [p.test_sd for p in config.performances]
    
    fig, axis = plt.subplots(figsize=(width, height))
    axis.bar(bar_x, bar_height, BAR_WIDTH, yerr=bar_error)

    axis.set_xticks(bar_x)
    axis.set_xticklabels([p.name for p in config.performances], ha="center")
    axis.set_ylabel("Accuracy [%]")
    axis.set_ylim((0, 100.0))
    
    if x_label is not None:
        axis.set_xlabel(x_label)
    
    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)

    fig.tight_layout(pad=0, rect=[0.0, 0.0, 1.0, 1.0])
    return fig, axis

def plot(configs, width=plot_settings.column_width, height=2.0):
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

    fig, axis = plt.subplots(figsize=(width, height))
    actors = axis.bar(bar_x, bar_height, BAR_WIDTH, yerr=bar_error, color=bar_colour)

    axis.set_xticks(group_x)
    axis.set_xticklabels([c.name for c in configs], ha="center")
    axis.set_ylabel("Accuracy [%]")
    axis.set_ylim((0, 100.0))

    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)

    fig.legend([mpatches.Rectangle(color=c, width=10, height=10, xy=(0,0)) for c in itervalues(colour_map)], 
               iterkeys(colour_map), loc="lower center", ncol=len(colour_map) if plot_settings.presentation or plot_settings.poster else 2)
    fig.tight_layout(pad=0, rect=[0.0, 0.2 if plot_settings.presentation or plot_settings.poster 
                                  else 0.2, 1.0, 1.0])
    return fig, axis

shd_configs = [Config("256 neurons", 
                     [("GeNN (eProp LSNN FF)", "shd_256_feedforward_100_epochs_*"), ("GeNN (eProp LSNN RC)", "shd_256_100_epochs_*")],
                     [Performance("PyTorch (BPTT LIF FF)", None, 74.0, 1.7), Performance("PyTorch (BPTT LIF RC)", None, 80.0, 2.0)]),
              Config("512 neurons", 
                     [("GeNN (eProp LSNN FF)", "shd_512_feedforward_100_epochs_*"), ("GeNN (eProp LSNN RC)", "shd_512_100_epochs_*")],
                     []),
              Config("1024 neurons", 
                     [("GeNN (eProp LSNN FF)", "shd_1024_feedforward_100_epochs_*"), ("GeNN (eProp LSNN RC)", "shd_1024_100_epochs_*")],
                     [])]

smnist_configs = [Config("256 neurons", 
                         [("GeNN (eProp LSNN RC)", "smnist_256_100_epochs_*")],
                         [Performance("TensorFlow (BPTT LSNN RC)", None, 96.4, 0.0), Performance("TensorFlow (BPTT LIF RC)", None, 63.3, 0.0)]),
                  Config("512 neurons", 
                         [("GeNN (eProp LSNN RC)", "smnist_512_100_epochs_*")],
                         []),
                  Config("1024 neurons", 
                         [("GeNN (eProp LSNN RC)", "smnist_1024_100_epochs_*")],
                         [])]

connectivities = [0.1, 0.05, 0.01]
sparse_configs = [Config(f"{in_con * 100:.0f}% input",
                         [(f"{rec_con * 100:.0f}% recurrent", f"shd_256_new_sparse_{in_con}_{rec_con}_epochs_*") for rec_con in connectivities],
                         [])
                  for in_con in connectivities]

experiments = [("Sparse Fixed", ""), ("Sparse Deep-R", "_deep_r_80_20")]
deep_r_configs = [Config(f"10% input\n{rec_con * 100:.0f}% recurrent",
                         [(e[0], f"shd_256_new_sparse_0.1_{rec_con}{e[1]}_epochs_*") for e in experiments],
                         [])
                  for rec_con in connectivities]

shd_fig, _ = plot(shd_configs)
smnist_fig, _ = plot(smnist_configs)
sparse_fig, _ = plot_simple(Config("",
                            [("100%", "shd_256_100_epochs_*"), 
                             ("10%", "shd_256_new_sparse_0.1_0.1_epochs_*"), 
                             ("5%", "shd_256_new_sparse_0.05_0.05_epochs_*"), 
                             ("1%", "shd_256_new_sparse_0.01_0.01_epochs_*")],
                            []), width=9.64, height=4.75, x_label="Connection density")

deep_r_fig, deep_r_axis = plot([Config("100%", [("Dense", "shd_256_100_epochs_*")], [])] + deep_r_configs,
                               width=9.64, height=5.0)

# Calculate p-values
for c in deep_r_configs:
    print(c.name)
    print(f"\t{c.performances[0].name} vs {c.performances[1].name}")
    print(f"\t{wilcoxon(c.performances[0].data, c.performances[1].data, alternative='less', mode='exact')}")
                  
if not plot_settings.presentation and not plot_settings.poster:
    shd_fig.savefig("../figures/shd_performance.pdf")
    smnist_fig.savefig("../figures/smnist_performance.pdf")
    sparse_fig.savefig("../figures/sparse_performance.pdf")
    deep_r_fig.savefig("../figures/deep_r_performance.pdf")

plt.show()
