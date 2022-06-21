import math
import sys
from glob import glob
from itertools import chain
from matplotlib import pyplot as plt
from plot_performance import plot, show_rewiring_legend

# Expand any remaining wildcards
directories = list(chain.from_iterable(glob(a) for a in sys.argv[1:]))

num_results = len(directories)
num_cols = int(math.ceil(math.sqrt(num_results)))
num_rows = int(math.ceil(float(num_results) / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, sharex="col", sharey="row")

best_trial = None
for i, (a, d) in enumerate(zip(axes.flatten(), directories)):
    a.set_title(d)
    max_train_performance, max_test_performance = plot(d, a)
    print(d)
    print("\tMax training performance: %f%%" % max_train_performance)
    print("\tMax testing performance: %f%%" % max_test_performance)

    if best_trial is None:
         best_trial = (d, max_test_performance)
    elif max_test_performance > best_trial[1]:
         best_trial = (d, max_test_performance)

    if i == 0:
        a.legend()
        show_rewiring_legend(a)

print("Best performance '%s': %f" % (best_trial[0], best_trial[1]))
if num_rows > 1:
    for i in range(num_rows):
        axes[i, 0].set_ylabel("Performance [%]")

    for i in range(num_cols):
        axes[-1, i].set_xlabel("Epoch")
else:
    axes[0].set_ylabel("Performance [%]")
    for i in range(num_cols):
        axes[i].set_xlabel("Epoch")
    
plt.show()
