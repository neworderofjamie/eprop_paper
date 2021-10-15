import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plot_settings

from glob import glob

def get_train_performance(path):
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

def get_test_performance(path):
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

# List of configurations to include
experiments = ["shd_256", "shd_256_feedforward", 
               "shd_512", "shd_512_feedforward", 
               "shd_1024", "shd_1024_feedforward"]

# Extract performance metrics
test_performance = []
train_performance = []
for e in experiments:
    test_performance.append([get_test_performance(e + "_100_epochs_%d" % i) for i in range(1, 4)])
    train_performance.append([get_train_performance(e + "_100_epochs_%d" % i) for i in range(1, 4)])

print(test_performance)
print(train_performance)


fig, axis = plt.subplots()
