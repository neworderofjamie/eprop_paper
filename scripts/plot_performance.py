import os
import numpy as np
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns 

from collections import namedtuple
from glob import glob

BAR_WIDTH = 1.0
BAR_PAD = 0.2
GROUP_PAD = 1.0

Performance = namedtuple("Performance", ["name", "train_mean", "train_sd",
                                         "test_mean", "test_sd"])

class Config(object):
    def __init__(self, name, experiment_files, references):
        self.name = name
        self.performances = references

        # Assert that there are some experiment files
        assert len(experiment_files) > 0
        
        # Loop through experiment files
        for name, file_prefix in experiment_files:
            # Find all directories containing runs of this experiment
            data_directories = glob(file_prefix + "_100_epochs_*")
            
            # Read test and training performance
            train_performance = [self._get_train_performance(d) for d in data_directories]
            test_performance = [self._get_test_performance(d) for d in data_directories]
            
            ## Add performance tuple
            self.performances.append(Performance(name, np.mean(train_performance), np.std(train_performance),
                                                 np.mean(test_performance), np.std(test_performance)))
    
    def _get_train_performance(self, path):
        # Load training data
        training_data = np.loadtxt(os.path.join(path, "performance.csv"), delimiter=",", skiprows=1)
        
        # Count epochs
        epochs = np.unique(training_data[:,0])

        num_trials = np.empty_like(epochs)
        num_correct = np.empty_like(epochs)

        for i, e in enumerate(epochs):
            epoch_mask = (training_data[:,0] == e)

            num_trials[i] = np.sum(training_data[epoch_mask,2])
            num_correct[i] = np.sum(training_data[epoch_mask,3])
        
        
        return 100.0 * (num_correct[-1] / num_trials[-1])

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


configs = [Config("256 neurons", 
                  [("eProp FF", "shd_256_feedforward"), ("eProp RC", "shd_256")],
                  [Performance("BPTT FF*", 97.0, 1.2, 74.0, 1.7), Performance("BPTT RC*", 98.0, 0.5, 80.0, 2.0)]),
           Config("512 neurons", 
                  [("eProp FF", "shd_512_feedforward"), ("eProp RC", "shd_512")],
                  []),
           Config("1024 neurons", 
                  [("eProp FF", "shd_1024_feedforward"), ("eProp RC", "shd_1024")],
                  [])]


pal = sns.color_palette()


bar_x = []
bar_height = []
bar_colour = []
bar_error = []
group_x = []

tick_label = []
tick_x = []

# Loop through configurations
last_x = 0.0
for c in configs:
    # Calculate centre of this group
    group_centre = (((len(c.performances) * 2) - 1) / 2.0) * (BAR_PAD + BAR_WIDTH)
    group_x.append(last_x + group_centre)
    
    # Loop through all performances achieved in this config
    for p in c.performances:
        # Add training bar
        bar_x.append(last_x)
        bar_height.append(p.train_mean)
        bar_colour.append(pal[0])
        bar_error.append(p.train_sd)
        
        # Add tick between bars
        tick_x.append(last_x + (BAR_WIDTH / 2))
        tick_label.append(p.name)
        
        # Add spacing between bars
        last_x += BAR_WIDTH
        
        # Add testing bar
        bar_x.append(last_x)
        bar_height.append(p.test_mean)
        bar_colour.append(pal[1])
        bar_error.append(p.test_sd)
        
        # Add spacing between bars
        last_x += (BAR_WIDTH + BAR_PAD)
    
    # Add extra padding between groups
    last_x += (GROUP_PAD - BAR_PAD)

fig, axis = plt.subplots()
actors = axis.bar(bar_x, bar_height, BAR_WIDTH, yerr=bar_error, color=bar_colour)

axis.set_xticks(tick_x)
axis.set_xticklabels(tick_label, ha="center")
axis.set_ylabel("Accuracy [%]")

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)

# Add neuron count labels
for x, c in zip(group_x, configs):
    axis.text(x, -10 if plot_settings.presentation else -12.0, c.name, ha="center",
              fontsize=15 if plot_settings.presentation else 9)

fig.legend([actors[0], actors[1]], ["Training", "Testing"], loc="lower center", ncol=2)
fig.tight_layout(pad=0, rect=[0.0, 0.075, 1.0, 1.0])
plt.show()
