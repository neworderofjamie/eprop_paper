import numpy as np

from collections import namedtuple
from copy import copy
from os import path
from struct import unpack

PreprocessedEvents = namedtuple("PreprocessedEvents", ["end_spikes", "spike_times"])

def get_start_spikes(end_spikes):
    start_spikes = np.empty_like(end_spikes)
    if end_spikes.ndim == 1:
        start_spikes[0] = 0
        start_spikes[1:] = end_spikes[:-1]
    else:
        start_spikes[0,0] = 0
        start_spikes[1:,0] = end_spikes[:-1,-1]
        start_spikes[:,1:] = end_spikes[:,:-1]

    return start_spikes

def concatenate_events(events):
    # Check that all stimuli are for same number of neurons
    assert all(len(e.end_spikes) == len(events[0].end_spikes) for e in events)

    # Extract seperate lists of each stimuli's end spike indices and spike times
    end_spikes, spike_times = zip(*events)

    # Make corresponding array of start spikes
    start_spikes = [get_start_spikes(e) for e in end_spikes]

    # Create empty array to hold spike times
    concat_spike_times = np.empty(sum(len(s) for s in spike_times))

    # End spikes are simply sum of the end spikes
    concat_end_spikes = np.sum(end_spikes, axis=0)

    # Loop through neurons
    start_idx = 0
    for i in range(len(concat_end_spikes)):
        # Loop through the start and end spike; 
        # and the spike times from each stimuli
        for s, e, t in zip(start_spikes, end_spikes, spike_times):
            # Copy block of spike times into place and advance start_idx
            num_spikes = e[i] - s[i]
            concat_spike_times[start_idx:start_idx + num_spikes] = t[s[i]:e[i]]
            start_idx += num_spikes

    return PreprocessedEvents(concat_end_spikes, concat_spike_times)

def batch_events(events, batch_size):
    # Check that there aren't more stimuli than batch size 
    # and that all stimuli are for same number of neurons
    num_neurons = len(events[0].end_spikes)
    assert len(events) <= batch_size
    assert all(len(e.end_spikes) == num_neurons for e in events)

    # Extract seperate lists of each stimuli's end spike indices and spike times
    end_spikes, spike_times = zip(*events)

    # Calculate cumulative sum of spikes counts across batch
    cum_spikes_per_stimuli = np.concatenate(([0], np.cumsum([len(s) for s in spike_times])))

    # Add this cumulative sum onto the end spikes array of each stimuli
    # **NOTE** zip will stop before extra cum_spikes_per_stimuli value
    batch_end_spikes = np.vstack([c + e
                                  for e, c in zip(end_spikes, cum_spikes_per_stimuli)])

    # If this isn't a full batch
    if len(events) < batch_size:
        # Create spike padding for remainder of batch
        spike_padding = np.ones((batch_size - len(events), num_neurons), dtype=int) * cum_spikes_per_stimuli[-1]

        # Stack onto end spikes
        batch_end_spikes = np.vstack((batch_end_spikes, spike_padding))

    # Concatenate together all spike times
    batch_spike_times = np.concatenate(spike_times)

    return PreprocessedEvents(batch_end_spikes, batch_spike_times)


def get_mnist(train):
    images_file = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    labels_file = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"

    # Open images file
    with open(path.join("mnist", images_file), "rb") as f:
        image_data = f.read()

        # Unpack header from first 16 bytes of buffer and verify
        magic, num_items, num_rows, num_cols = unpack(">IIII", image_data[:16])
        assert magic == 2051
        assert num_rows == 28
        assert num_cols == 28

        # Convert remainder of buffer to numpy bytes
        image_data = np.frombuffer(image_data[16:], dtype=np.uint8)

        # Reshape data into individual (flattened) images
        image_data = np.reshape(image_data, (num_items, 28 * 28))

    # Open labels file
    with open(path.join("mnist", labels_file), "rb") as f:
        label_data = f.read()

        # Unpack header from first 8 bytes of buffer and verify
        magic, num_items = unpack(">II", label_data[:8])
        assert magic == 2049

        # Convert remainder of buffer to numpy bytes
        label_data = np.frombuffer(label_data[8:], dtype=np.uint8)
        assert label_data.shape == (image_data.shape[0],)
    return image_data, label_data

class DataIter:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iteration = 0
        self.indices = np.arange(0, self.data_loader.length, dtype=int)
        if self.data_loader.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        dl = self.data_loader
        if self.iteration >= dl.length:
            raise StopIteration
        elif dl.batch_size == 1:
            idx = self.indices[self.iteration]
            self.iteration += 1
            return ([dl._preprocessed_events[idx]], [dl._labels[idx]])
        else:
            # Get start and end of batch (end might be past end of indices)
            begin = self.iteration
            end = self.iteration + dl.batch_size

            # Get indices and thus slice of data
            inds = self.indices[begin:end]

            # Add number of indices to iteration count
            # (will take into account size of dataset)
            self.iteration += len(inds)

            # Return list of data
            return ([dl._preprocessed_events[i] for i in inds],
                    dl._labels[inds])

class SpikingDataLoader:
    def __init__(self, dataset, shuffle=False, batch_size=1, 
                 sensor_size=None, dataset_slice=slice(0,None), time_scale=1.0 / 1000.0):
        # Build list of dataset indices in our slice
        full_length = len(dataset)
        slice_indices = list(range(full_length)[dataset_slice])

        self.length = len(slice_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.time_scale = time_scale

        # Zero maximums
        self.max_stimuli_time = 0.0
        self.max_spikes_per_stimuli = 0

        # Allocate numpy array for labels
        self._labels = np.empty(self.length, dtype=int)
        self._preprocessed_events = []

        # Override sensor size if required
        sensor_size = (sensor_size if sensor_size is not None 
                       else dataset.sensor_size)

        # Loop through slice of dataset
        for i, s in enumerate(slice_indices):
            events, label = dataset[s]

            # Store label
            self._labels[i] = label

            # Preprocess events 
            preproc_events = self._preprocess(events, dataset.ordering, 
                                              sensor_size)

            # Update max spike times and max spikes per stimuli
            self.max_stimuli_time = max(self.max_stimuli_time, 
                                        np.amax(preproc_events.spike_times))
            self.max_spikes_per_stimuli = max(self.max_spikes_per_stimuli,
                                              len(preproc_events.spike_times))
            # Add events to list
            self._preprocessed_events.append(preproc_events)

    def __iter__(self):
        return DataIter(self)

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))

    def _preprocess(self, events, ordering, sensor_size):
        # Calculate cumulative sum of each neuron's spike count
        num_input_neurons = np.product(sensor_size) 

        # Check dataset datatype includes time and polarity
        assert "t" in ordering
        assert "p" in ordering

        # If sensor has single polarity
        if sensor_size[2] == 1:
            # If sensor is 2D, flatten x and y into event IDs
            if ("x" in ordering) and ("y" in ordering):
                spike_event_ids = events["x"] + (events["y"] * sensor_size[1])
            # Otherwise, if it's 1D, simply use X
            elif "x" in ordering:
                spike_event_ids = events["x"]
            else:
                raise "Only 1D and 2D sensors supported"
        # Otherwise
        else:
            # If sensor is 2D, flatten x, y and p into event IDs
            if ("x" in ordering) and ("y" in ordering):
                spike_event_ids = (events["p"] +
                                   (events["x"] * sensor_size[2]) + 
                                   (events["y"] * sensor_size[1] * sensor_size[2]))
            # Otherwise, if it's 1D, flatten x and p into event IDs
            elif "x" in ordering:
                spike_event_ids = events["p"] + (events["x"] * sensor_size[2])
            else:
                raise "Only 1D and 2D sensors supported"

        end_spikes = np.cumsum(np.bincount(spike_event_ids, 
                                           minlength=num_input_neurons))
        assert len(end_spikes) == num_input_neurons

        # Sort events first by neuron id and then by time and use to order spike times
        spike_times = events["t"][np.lexsort((events["t"], spike_event_ids))]

        # Return end spike indices and spike times (converted to floating point ms)
        return PreprocessedEvents(end_spikes, spike_times * self.time_scale)

class LinearLatencyEncoder(object):
    def __init__(self, thresh, max_stimuli_time):
        self.thresh = thresh
        self.max_stimuli_time = max_stimuli_time

    def __call__(self, image_data, label_data, slice_indices):
        # Loop through slice of dataset
        labels = np.empty(image_data.shape[0], dtype=int)
        preprocessed_events = []
        max_spikes_per_stimuli = 0
        max_stimuli_time = 0.0
        scale = self.max_stimuli_time / 255.0
        for i, s in enumerate(slice_indices):
            # Store label
            labels[i] = label_data[s]

            # Get boolean mask of spiking neurons
            spike_vector = image_data[s] > self.thresh

            # Take cumulative sum to get end spikes
            end_spikes = np.cumsum(spike_vector)

            # Extract values of spiking pixels
            spiking_pixels = image_data[s,spike_vector]

            # Update max spikes per stimuli
            max_spikes_per_stimuli = max(max_spikes_per_stimuli, len(spiking_pixels))

            # Calculate spike times
            spike_times = (255 - spiking_pixels) * scale

            # Update max stimuli time
            max_stimuli_time = max(max_stimuli_time, np.amax(spike_times))

            # Add preprocessed_events tuple to
            preprocessed_events.append(PreprocessedEvents(end_spikes, spike_times))

        return (max_stimuli_time, max_spikes_per_stimuli,
                labels, preprocessed_events)

class LogLatencyEncoder(object):
    def __init__(self, tau_eff, thresh, max_stimuli_time=None):
        self.tau_eff = tau_eff
        self.thresh = thresh
        self.max_stimuli_time = max_stimuli_time

    def __call__(self, image_data, label_data, slice_indices):
        # Loop through slice of dataset
        labels = np.empty(image_data.shape[0], dtype=int)
        preprocessed_events = []
        max_stimuli_time = 0.0
        max_spikes_per_stimuli = 0
        for i, s in enumerate(slice_indices):
            # Store label
            labels[i] = label_data[s]

            # Get boolean mask of spiking neurons
            spike_vector = image_data[s] > self.thresh

            # Take cumulative sum to get end spikes
            end_spikes = np.cumsum(spike_vector)

            # Extract values of spiking pixels
            spiking_pixels = image_data[s,spike_vector]

            # Update max spikes per stimuli
            max_spikes_per_stimuli = max(max_spikes_per_stimuli, len(spiking_pixels))

            # Calculate spike times
            spike_times = self.tau_eff * np.log(spiking_pixels / (spiking_pixels - self.thresh))

            # Update max stimuli time
            max_stimuli_time = max(max_stimuli_time, np.amax(spike_times))

            # Add preprocessed_events tuple to 
            preprocessed_events.append(PreprocessedEvents(end_spikes, spike_times))

        # If max stimuli time is specified, override calculated value
        if self.max_stimuli_time is not None:
            max_stimuli_time = self.max_stimuli_time

        return (max_stimuli_time, max_spikes_per_stimuli, 
                labels, preprocessed_events)

class ImageDataLoader:
    def __init__(self, dataset, shuffle=False, batch_size=1, 
                 encoder=None, dataset_slice=slice(0,None)):
        # Build list of dataset indices in our slice
        full_length = dataset[0].shape[0]
        slice_indices = list(range(full_length)[dataset_slice])

        self.length = len(slice_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        (self.max_stimuli_time, self.max_spikes_per_stimuli, 
         self._labels, self._preprocessed_events) = encoder(dataset[0], dataset[1], 
                                                            slice_indices)

    def __iter__(self):
        return DataIter(self)

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))
