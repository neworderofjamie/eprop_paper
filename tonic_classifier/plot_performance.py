from glob import glob
import numpy as np
import os
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from tonic_classifier_parser import parse_arguments

def label_rewiring_axis(axis):
    axis_siblings = axis.get_shared_x_axes().get_siblings(axis)
    if len(axis_siblings) > 1:
        axis_siblings[0].set_ylabel("Num rewirings")

def plot(output_directory, axis):
    # Load training data
    training_data = np.loadtxt(os.path.join(output_directory, "performance.csv"), delimiter=",", skiprows=1)
    
    # Count epochs
    epochs = np.unique(training_data[:,0])

    num_trials = np.empty_like(epochs)
    num_correct = np.empty_like(epochs)
    num_rewirings = [np.empty_like(epochs) for i in range(training_data.shape[1] - 4)]
    for i, e in enumerate(epochs):
        epoch_mask = (training_data[:,0] == e)

        num_trials[i] = np.sum(training_data[epoch_mask,2])
        num_correct[i] = np.sum(training_data[epoch_mask,3])
        
        for j in range(training_data.shape[1] - 4):
            num_rewirings[j][i] = np.sum(training_data[epoch_mask, j + 4])
    
    if len(num_rewirings) > 0:
        rewiring_axis = axis.twinx()
        rewiring_axis.set_ylim((0, 1000))
    
    max_train_performance = 100.0 * (num_correct[-1] / num_trials[-1])

    train_actor = axis.plot(100.0 * num_correct / num_trials, label="Training")[0]
    axis.axhline(max_train_performance, linestyle="--", color=train_actor.get_color())
    axis.annotate("%0.2f%%" % max_train_performance, (0.0, max_train_performance), 
                  ha="right", va="center", color=train_actor.get_color())
    
    for i, r in enumerate(num_rewirings):
        rewiring_axis.plot(r, label=f"Rewiring {i}")

    # Find evaluation files, sorting numerically
    evaluate_files = list(sorted(glob(os.path.join(output_directory, "performance_evaluate_*.csv")),
                                key=lambda x: int(os.path.basename(x)[21:-4])))

    # Loop through evaluate files
    test_epoch = []
    test_performance = []
    for e in evaluate_files:
        # Extract epoch number
        epoch = int(os.path.basename(e)[21:-4])

        # Load file
        test_data = np.loadtxt(e, delimiter=",", skiprows=1)

        # Calculate performance
        num_trials = np.sum(test_data[:,1])
        num_correct = np.sum(test_data[:,2])

        # Add to list
        test_performance.append(100.0 * num_correct / num_trials)
        test_epoch.append(epoch)

    if len(test_performance) > 0:
        max_epoch = max(max(test_epoch), np.amax(epochs))
    else:
        max_epoch = np.amax(epochs)

    axis.set_xlim((0, max_epoch))
    axis.set_ylim((0, 100))

    if len(test_performance) > 0:
        # Plot
        test_actor = axis.plot(test_epoch, test_performance, label="Testing")[0]

        max_test_performance = test_performance[min(int(max_epoch), len(test_performance) - 1)]
        axis.axhline(max_test_performance, linestyle="--", color=test_actor.get_color())
        axis.annotate("%0.2f%%" % max_test_performance, (max_epoch, max_test_performance), 
                      ha="left", va="center", color=test_actor.get_color())
        return max_train_performance, max_test_performance
    else:
        return max_train_performance, 0.0

if __name__ == "__main__":
    # Parse command line
    name_suffix, output_directory, _ = parse_arguments(description="Plot eProp classifier performance")
    
    fig, axis = plt.subplots()
    max_train_performance, max_test_performance = plot(output_directory, axis)
    print("Max training performance: %f%%" % max_train_performance)
    print("Max testing performance: %f%%" % max_test_performance)
    axis.legend()
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Performance [%]")
    label_rewiring_axis(axis)
        
    axis.set_title(name_suffix)
    plt.show()
