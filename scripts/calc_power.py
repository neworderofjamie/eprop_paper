from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# Get power trace files
power_traces = list(glob("power_traces/*.txt"))

num_axis = int(np.ceil(np.sqrt(len(power_traces))))
fig, axes = plt.subplots(num_axis, num_axis)

for trace, ax in zip(power_traces, axes.flatten()):
    power = []
    with open(trace, "r") as f:
        lines=f.readlines()
        for line in lines:
            try:
                power.append(float(line.split(',')[1].split('W')[0]))
            except:
                print("Error parsing '%s'" % line)
    
    
    # Convert to numpy
    power = np.asarray(power)
    
    # Calculate gradient
    gradient = np.gradient(power)
    
    # Find maximum and minimum gradient which should represent start and end of experiment
    experiment_start = np.argmax(gradient)
    experiment_end = np.argmin(gradient)
    assert experiment_start < experiment_end
    
    # Calcualte average power
    static_power = np.average(power[:experiment_start])
    total_power = np.average(power[experiment_start + 1:experiment_end])
    print("%s: Static power:%f W, Total power:%f W" % (trace, static_power, total_power))
    
    # Plot power
    ax.plot(power)
    ax.axhline(static_power, linestyle="--", color="gray")
    ax.axhline(total_power, linestyle="--", color="gray")
    ax.axvline(experiment_start, linestyle="--", color="gray")
    ax.axvline(experiment_end, linestyle="--", color="gray")
    ax.set_title(trace)
    ax.set_ylabel("Power [W]")
    ax.set_xlabel("Sample")

plt.show()