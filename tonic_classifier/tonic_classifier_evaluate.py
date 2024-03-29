import csv
import numpy as np
import os
import tonic
import matplotlib.pyplot as plt
import random
import subprocess

from argparse import ArgumentParser
from time import perf_counter, sleep
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from tonic_classifier_parser import parse_arguments
import dataloader

# Eprop imports
#import eprop

# Build command line parse
parser = ArgumentParser(add_help=False)
parser.add_argument("--timing", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--warmup", action="store_true")
parser.add_argument("--record-power", action="store_true")
parser.add_argument("--backend")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--trained-epoch", type=int, default=49)
parser.add_argument("--cuda-visible-devices", action="store_true")
parser.add_argument("--hold-back-validate", type=int, default=None)

name_suffix, output_directory, args = parse_arguments(parser, description="Evaluate eProp classifier")
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Seed RNG, leaving random to match GeNN behaviour if seed is zero
np.random.seed(None if args.seed == 0 else args.seed)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def write_spike_file(filename, data):
    np.savetxt(filename, np.column_stack(data), fmt=["%f","%d"], 
               delimiter=",", header="Time [ms], Neuron ID")

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
recurrent_lif_model = genn_model.create_custom_neuron_class(
    "recurrent_lif",
    param_names=["TauM", "Vthresh", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar")],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],

    sim_code="""
    $(V) = ($(Alpha) * $(V)) + $(Isyn);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(V) -= $(Vthresh);
    """,
    threshold_condition_code="""
    $(RefracTime) <= 0.0 && $(V) >= $(Vthresh)
    """,
    is_auto_refractory_required=False)

recurrent_alif_model = genn_model.create_custom_neuron_class(
    "recurrent_alif",
    param_names=["TauM", "TauAdap", "Vthresh", "TauRefrac", "Beta"],
    var_name_types=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar")],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))())],

    sim_code="""
    $(V) = ($(Alpha) * $(V)) + $(Isyn);
    $(A) *= $(Rho);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(V) -= $(Vthresh);
    $(A) += 1.0;
    """,
    threshold_condition_code="""
    $(RefracTime) <= 0.0 && $(V) >= ($(Vthresh) + ($(Beta) * $(A)))
    """,
    is_auto_refractory_required=False)

output_classification_model = genn_model.create_custom_neuron_class(
    "output_classification",
    param_names=["TauOut"],
    var_name_types=[("Y", "scalar"), ("YSum", "scalar"), ("B", "scalar", VarAccess_READ_ONLY)],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],

    sim_code="""
    // Reset YSum at start of each batch
    if($(t) == 0.0) {
        $(YSum) = 0.0;
    }

    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);

    $(YSum) += $(Y);
    """,
    is_auto_refractory_required=False)

# Create dataset
sensor_size = None
encoder = None
spiking = True
num_outputs = None
num_input_neurons = None
time_scale = 1.0 / 1000.0
if args.dataset == "shd":
    transformations = []
    if args.crop_time:
        transformations.append(tonic.transforms.CropTime(max=args.crop_time * 1000.0))
    transformations.append(dataloader.EventsToGrid(tonic.datasets.SHD.sensor_size, args.dt * 1000))
    dataset = tonic.datasets.SHD(save_to='./data', train=args.hold_back_validate is not None)
    sensor_size = dataset.sensor_size
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', duplicate=False, num_neurons=79, 
                                    train=args.hold_back_validate is not None, download=not args.no_download_dataset)
    sensor_size = dataset.sensor_size
elif args.dataset == "dvs_gesture":
    transform = tonic.transforms.Compose([
        tonic.transforms.Downsample(spatial_factor=0.25)])
    dataset = tonic.datasets.DVSGesture(save_to='./data', train=args.hold_back_validate is not None,
                                        transform=transform)
    sensor_size = (32, 32, 2)
    time_scale = 1.0
elif args.dataset == "mnist":
    import mnist
    
    num_outputs = 10
    if args.hold_back_validate is None:
        dataset = (mnist.test_images(), mnist.test_labels())
    else:
        dataset = (mnist.train_images(), mnist.train_labels())
    encoder = dataloader.LogLatencyEncoder(args.log_latency_tau, args.log_latency_threshold, 100)
    spiking = False
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)


# If we're using held back training set data, only use this part of dataset
dataset_slice = slice(None) if args.hold_back_validate is None else slice(-args.hold_back_validate, None)

# Create loader
start_processing_time = perf_counter()
if spiking:
    num_outputs = len(dataset.classes)
    num_input_neurons = np.product(sensor_size) 
    data_loader = dataloader.SpikingDataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                                               sensor_size=sensor_size, dataset_slice=dataset_slice,
                                               time_scale=time_scale)
else:
    assert encoder is not None
    num_input_neurons = np.product(dataset[0].shape[1:])
    data_loader = dataloader.ImageDataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                                             encoder=encoder, dataset_slice=dataset_slice)
data_loader.max_stimuli_time = 1000.0

end_process_time = perf_counter()
print("Data processing time:%f ms" % ((end_process_time - start_processing_time) * 1000.0))

# Round up to power-of-two
# **NOTE** not really necessary for evaluation - could slice up weights
num_output_neurons = int(2**(np.ceil(np.log2(num_outputs))))

# Flags for simplifying logic
input_recurrent_sparse = (args.input_recurrent_sparsity != 1.0)
recurrent_recurrent_sparse = (args.recurrent_recurrent_sparsity != 1.0)
input_recurrent_deep_r = args.deep_r and input_recurrent_sparse
recurrent_recurrent_deep_r = args.deep_r and recurrent_recurrent_sparse
input_recurrent_inh_deep_r = input_recurrent_deep_r and args.input_recurrent_inh_sparsity is not None

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Recurrent population
recurrent_alif_params = {"TauM": 20.0, "TauAdap": 2000.0, "Vthresh": 0.6, "TauRefrac": 5.0, "Beta": 0.0174}
recurrent_alif_vars = {"V": 0.0, "A": 0.0, "RefracTime": 0.0}
recurrent_lif_params = {"TauM": 20.0, "Vthresh": 0.6, "TauRefrac": 5.0}
recurrent_lif_vars = {"V": 0.0, "RefracTime": 0.0}

# Output population
output_params = {"TauOut": 20.0}

if args.no_bias:
    output_vars = {"Y": 0.0, "YSum": 0.0, "B": 0.0}
else:
    output_vars = {"Y": 0.0, "YSum": 0.0, "B": np.load(os.path.join(output_directory, "b_output_%u.npy" % args.trained_epoch))}

# (For now) check that there aren't both LIF and ALIF recurrent neurons
assert not (args.num_recurrent_alif > 0 and args.num_recurrent_lif > 0)

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
if args.num_recurrent_alif > 0:
    # Input->recurrent synapse parameters
    input_recurrent_alif_vars = {"g": np.load(os.path.join(output_directory, "g_input_recurrent_%u.npy" % args.trained_epoch))}
    
    if input_recurrent_sparse:
        input_recurrent_alif_inds = np.load(os.path.join(output_directory, "ind_input_recurrent_%u.npy" % args.trained_epoch))
    
    if input_recurrent_deep_r:
        assert np.all(input_recurrent_alif_vars["g"] >= 0.0)
    
    if input_recurrent_inh_deep_r:
        input_recurrent_alif_inh_vars = {"g": -np.load(os.path.join(output_directory, "g_input_recurrent_inh_%u.npy" % args.trained_epoch))}
        input_recurrent_alif_inh_inds = np.load(os.path.join(output_directory, "ind_input_recurrent_inh_%u.npy" % args.trained_epoch))
        
        assert np.all(input_recurrent_alif_inh_vars["g"] <= 0.0)

    # Recurrent->output synapse parameters
    recurrent_alif_output_vars = {"g": np.load(os.path.join(output_directory, "g_recurrent_output_%u.npy" % args.trained_epoch))}

if args.num_recurrent_lif > 0:
    # Input->recurrent synapse parameters
    input_recurrent_lif_vars = {"g": np.load(os.path.join(output_directory, "g_input_recurrent_lif_%u.npy" % args.trained_epoch))}

    if input_recurrent_sparse:
        assert np.all(input_recurrent_lif_vars["g"] >= 0.0)
        input_recurrent_lif_inds = np.load(os.path.join(output_directory, "ind_input_recurrent_lif_%u.npy" % args.trained_epoch))

    if input_recurrent_deep_r:
        assert np.all(input_recurrent_lif_vars["g"] >= 0.0)
    
    if input_recurrent_inh_deep_r:
        input_recurrent_lif_inh_vars = {"g": -np.load(os.path.join(output_directory, "g_input_recurrent_lif__inh_%u.npy" % args.trained_epoch))}
        input_recurrent_lif_inh_inds = np.load(os.path.join(output_directory, "ind_input_recurrent_lif_inh_%u.npy" % args.trained_epoch))
        
        assert np.all(input_recurrent_lif_inh_vars["g"] <= 0.0)


    # Recurrent->output synapse parameters
    recurrent_lif_output_vars = {"g": np.load(os.path.join(output_directory, "g_recurrent_lif_output_%u.npy" % args.trained_epoch))}

# Recurrent->recurrent synapse parameters
if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_vars = {"g": np.load(os.path.join(output_directory, "g_recurrent_recurrent_%u.npy" % args.trained_epoch))}

        if recurrent_recurrent_sparse:
            recurrent_alif_recurrent_alif_inds = np.load(os.path.join(output_directory, "ind_recurrent_recurrent_%u.npy" % args.trained_epoch))

        if recurrent_recurrent_deep_r:
            assert np.all(recurrent_alif_recurrent_alif_vars["g"] >= 0.0)
            num_excitatory = round(args.recurrent_excitatory_fraction * args.num_recurrent_alif)
            inhibitory_mask = (recurrent_alif_recurrent_alif_inds[0] >= num_excitatory)
            recurrent_alif_recurrent_alif_vars["g"][inhibitory_mask] *= -1.0

    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_vars = {"g": np.load(os.path.join(output_directory, "g_recurrent_lif_recurrent_lif_%u.npy" % args.trained_epoch))}

        if recurrent_recurrent_sparse:
            recurrent_lif_recurrent_lif_inds = np.load(os.path.join(output_directory, "ind_recurrent_lif_recurrent_lif_%u.npy" % args.trained_epoch))

        if recurrent_recurrent_deep_r:
            assert np.all(recurrent_lif_recurrent_lif_vars["g"] >= 0.0)
            num_excitatory = round(args.recurrent_excitatory_fraction * args.num_recurrent_lif)
            inhibitory_mask = (recurrent_lif_recurrent_lif_inds[0] >= num_excitatory)
            recurrent_alif_recurrent_lif_vars["g"][inhibitory_mask] *= -1.0

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
if args.backend == "CUDA":
    kwargs = {"selectGPUByDeviceID": True, "deviceSelectMethod": DeviceSelect_MANUAL} if args.cuda_visible_devices else {}
elif args.backend == "SingleThreadedCPU":
    kwargs = {"userCxxFlagsGNU": "-march=native"}
else:
    kwargs = {}
model = genn_model.GeNNModel("float", "%s_tonic_classifier_evaluate_%s" % (args.dataset, name_suffix), 
                             backend=args.backend, **kwargs)
model.dT = args.dt
model.timing_enabled = args.timing
model.batch_size = args.batch_size
model._model.set_seed(args.seed)
model._model.set_fuse_postsynaptic_models(True)

# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})

if args.num_recurrent_alif > 0:
    recurrent_alif = model.add_neuron_population("RecurrentALIF", args.num_recurrent_alif, recurrent_alif_model,
                                                 recurrent_alif_params, recurrent_alif_vars)
    recurrent_alif.spike_recording_enabled = args.record

if args.num_recurrent_lif > 0:
    recurrent_lif = model.add_neuron_population("RecurrentLIF", args.num_recurrent_lif, recurrent_lif_model,
                                                recurrent_lif_params, recurrent_lif_vars)
    recurrent_lif.spike_recording_enabled = args.record

output = model.add_neuron_population("Output", num_output_neurons, output_classification_model,
                                     output_params, output_vars)

# Allocate memory for input spikes and labels
input.set_extra_global_param("spikeTimes", np.zeros(args.batch_size * data_loader.max_spikes_per_stimuli, dtype=np.float32))

# Turn on recording
input.spike_recording_enabled = args.record

# Add synapse populations
input_recurrent_matrix_type = ("SPARSE_INDIVIDUALG" if input_recurrent_sparse
                               else "DENSE_INDIVIDUALG")
if args.num_recurrent_alif > 0:
    input_recurrent_alif = model.add_synapse_population(
        "InputRecurrentALIF", input_recurrent_matrix_type, NO_DELAY,
        input, recurrent_alif,
        "StaticPulse", {}, input_recurrent_alif_vars, {}, {},
        "DeltaCurr", {}, {})
    if input_recurrent_sparse:
        input_recurrent_alif.set_sparse_connections(input_recurrent_alif_inds[0],
                                                    input_recurrent_alif_inds[1])
        
    if input_recurrent_inh_deep_r:
        input_recurrent_alif_inh = model.add_synapse_population(
            "InputRecurrentALIFInh", input_recurrent_matrix_type, NO_DELAY,
            input, recurrent_alif,
            "StaticPulse", {}, input_recurrent_alif_inh_vars, {}, {},
            "DeltaCurr", {}, {})
        input_recurrent_alif_inh.set_sparse_connections(input_recurrent_alif_inh_inds[0],
                                                        input_recurrent_alif_inh_inds[1])
        
    recurrent_alif_output = model.add_synapse_population(
        "RecurrentALIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
        recurrent_alif, output,
        "StaticPulse", {}, recurrent_alif_output_vars, {}, {},
        "DeltaCurr", {}, {})
        
if args.num_recurrent_lif > 0:
    input_recurrent_lif = model.add_synapse_population(
        "InputRecurrentLIF", input_recurrent_matrix_type, NO_DELAY,
        input, recurrent_lif,
        "StaticPulse", {}, input_recurrent_lif_vars, {}, {},
        "DeltaCurr", {}, {})
    
    if input_recurrent_sparse:
        input_recurrent_lif.set_sparse_connections(input_recurrent_lif_inds[0],
                                                   input_recurrent_lif_inds[1])
    
    if input_recurrent_inh_deep_r:
        input_recurrent_lif_inh = model.add_synapse_population(
            "InputRecurrentLIFInh", input_recurrent_matrix_type, NO_DELAY,
            input, recurrentalif,
            "StaticPulse", {}, input_recurrent_lif_inh_vars, {}, {},
            "DeltaCurr", {}, {})
        input_recurrent_lif_inh.set_sparse_connections(input_recurrent_lif_inh_inds[0],
                                                       input_recurrent_lif_inh_inds[1])

    recurrent_lif_output = model.add_synapse_population(
        "RecurrentLIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
        recurrent_lif, output,
        "StaticPulse", {}, recurrent_lif_output_vars, {}, {},
        "DeltaCurr", {}, {})
    

if not args.feedforward:
    recurrent_recurrent_matrix_type = ("SPARSE_INDIVIDUALG" if recurrent_recurrent_sparse
                                       else "DENSE_INDIVIDUALG")

    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif = model.add_synapse_population(
            "RecurrentALIFRecurrentALIF", recurrent_recurrent_matrix_type, NO_DELAY,
            recurrent_alif, recurrent_alif,
            "StaticPulse", {}, recurrent_alif_recurrent_alif_vars, {}, {},
            "DeltaCurr", {}, {})
        
        if recurrent_recurrent_sparse:
            recurrent_alif_recurrent_alif.set_sparse_connections(recurrent_alif_recurrent_alif_inds[0],
                                                                 recurrent_alif_recurrent_alif_inds[1])
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif = model.add_synapse_population(
            "RecurrentLIFRecurrentLIF", recurrent_recurrent_matrix_type, NO_DELAY,
            recurrent_lif, recurrent_lif,
            "StaticPulse", {}, recurrent_lif_recurrent_lif_vars, {}, {},
            "DeltaCurr", {}, {})
        
        if recurrent_recurrent_sparse:
            recurrent_lif_recurrent_lif.set_sparse_connections(recurrent_lif_recurrent_lif_inds[0],
                                                               recurrent_lif_recurrent_lif_inds[1])

# Build and load model
stimuli_timesteps = int(np.ceil(data_loader.max_stimuli_time / args.dt))
model.build()
model.load(num_recording_timesteps=stimuli_timesteps)

# Get views
input_neuron_start_spike = input.vars["startSpike"].view
input_neuron_end_spike = input.vars["endSpike"].view
input_spike_times_view = input.extra_global_params["spikeTimes"].view

output_y_sum_view = output.vars["YSum"].view

if args.reset_neurons:
    assert args.num_recurrent_alif == 0
    recurrent_lif_v_view = recurrent_lif.vars["V"].view
    output_y_view = output.vars["Y"].view

# Open file
performance_file = open(os.path.join(output_directory, "performance_evaluate_%u.csv" % args.trained_epoch), "w")
performance_csv = csv.writer(performance_file, delimiter=",")
performance_csv.writerow(("Batch", "Num trials", "Number correct"))

# If we should record power, launch nvidia-smi process
if args.record_power:
    power_trace_filename = os.path.join(output_directory, "power_trace.txt")
    call = "nvidia-smi -i 0 -lms 10 --format=csv,noheader --query-gpu=timestamp,power.draw -f %s" % power_trace_filename
    process = subprocess.Popen(call.split())
    sleep(5)

# If we should warmup the state of the network
if args.warmup:
    # Loop through batches of (pre-processed) data
    data_iter = iter(data_loader)
    for events, _ in data_iter:
        # Transform data into batch
        batched_data = dataloader.batch_events(events, args.batch_size)

        # Reset time
        model.timestep = 0
        model.t = 0.0

        # Check that spike times will fit in view, copy them and push them
        assert len(batched_data.spike_times) <= len(input_spike_times_view)
        input_spike_times_view[0:len(batched_data.spike_times)] = batched_data.spike_times
        input.push_extra_global_param_to_device("spikeTimes")

        # Calculate start and end spike indices
        input_neuron_end_spike[:] = batched_data.end_spikes
        input_neuron_start_spike[:] = dataloader.get_start_spikes(batched_data.end_spikes)
        input.push_var_to_device("startSpike")
        input.push_var_to_device("endSpike")

        # Loop through timesteps
        for i in range(stimuli_timesteps):
            model.step_time()

total_num = 0;
total_num_correct = 0
start_time = perf_counter()
batch_times = []
# Loop through batches of (pre-processed) data
data_iter = iter(data_loader)
for batch_idx, (events, labels) in enumerate(data_iter):
    print("Batch %u" % batch_idx)
    batch_start_time = perf_counter()

    # Transform data into batch
    batched_data = dataloader.batch_events(events, args.batch_size)

    # Reset time
    model.timestep = 0
    model.t = 0.0

    # Check that spike times will fit in view, copy them and push them
    assert len(batched_data.spike_times) <= len(input_spike_times_view)
    input_spike_times_view[0:len(batched_data.spike_times)] = batched_data.spike_times
    input.push_extra_global_param_to_device("spikeTimes")

    # Calculate start and end spike indices
    input_neuron_end_spike[:] = batched_data.end_spikes
    input_neuron_start_spike[:] = dataloader.get_start_spikes(batched_data.end_spikes)
    input.push_var_to_device("startSpike")
    input.push_var_to_device("endSpike")

    # Loop through timesteps
    num_correct = 0
    classification_output = np.zeros((len(labels), num_outputs))
    for i in range(stimuli_timesteps):
        model.step_time()

    # Pull sum of outputs from device
    output.pull_var_from_device("YSum")

    # If maximum output matches label, increment counter
    if args.batch_size == 1:
        num_correct += np.sum(np.argmax(output_y_sum_view) == labels)
    else:
        num_correct += np.sum(np.argmax(output_y_sum_view[:len(labels),:], axis=1) == labels)

    print("\t%u / %u correct = %f %%" % (num_correct, len(labels), 100.0 * num_correct / len(labels)))
    total_num += len(labels)
    total_num_correct += num_correct

    performance_csv.writerow((batch_idx, len(labels), num_correct))
    performance_file.flush()

    if args.reset_neurons:
        recurrent_lif_v_view[:] = 0.0
        output_y_view[:] = 0.0
        recurrent_lif.push_var_to_device("V")
        output.push_var_to_device("Y")
    if args.record:
        # Download recording data
        model.pull_recording_buffers_from_device()

        # Write spikes
        for i, s in enumerate(input.spike_recording_data):
            write_spike_file(os.path.join(output_directory, "input_spikes_%u_%u.csv" % (batch_idx, i)), s)

        if args.num_recurrent_alif > 0:
            for i, s in enumerate(recurrent_alif.spike_recording_data):
                write_spike_file(os.path.join(output_directory, "recurrent_spikes_%u_%u_%u.csv" % (args.trained_epoch, batch_idx, i)), s)
        if args.num_recurrent_lif > 0:
            for i, s in enumerate(recurrent_lif.spike_recording_data):
                write_spike_file(os.path.join(output_directory, "recurrent_lif_spikes_%u_%u_%u.csv" % (args.trained_epoch, batch_idx, i)), s)

    batch_end_time = perf_counter()
    batch_times.append((batch_end_time - batch_start_time) * 1000.0)
    print("\t\tTime:%f ms" % batch_times[-1])

end_time = perf_counter()
print("%u / %u correct = %f %%" % (total_num_correct, total_num, 100.0 * total_num_correct / total_num))
print("Time:%f ms" % ((end_time - start_time) * 1000.0))
print("Average batch time: %f ms" % np.average(batch_times))

performance_file.close()
if args.timing:
    print("Init: %f" % model.init_time)
    print("Init sparse: %f" % model.init_sparse_time)
    print("Neuron update: %f" % model.neuron_update_time)
    print("Presynaptic update: %f" % model.presynaptic_update_time)

if args.record_power:
    sleep(5)
    process.terminate()
