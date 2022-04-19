import numpy as np
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

# Eprop imports
import eprop
from deep_r import DeepR

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

C_L1 = 0.01

WEIGHT_0 = 1.0

NUM_INPUT = 20
NUM_RECURRENT = 256
NUM_OUTPUT = 3

parser = ArgumentParser(description="Pattern recognition")
parser.add_argument("--deep-r", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--input-recurrent-sparsity", type=float, default=1.0)
parser.add_argument("--recurrent-recurrent-sparsity", type=float, default=1.0)

args = parser.parse_args()

sparse_input_recurrent = (args.input_recurrent_sparsity != 1.0)
sparse_recurrent_recurrent = (args.recurrent_recurrent_sparsity != 1.0)
input_recurrent_deep_r = args.deep_r and sparse_input_recurrent 
recurrent_recurrent_deep_r = args.deep_r and sparse_recurrent_recurrent 

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
input_model = genn_model.create_custom_neuron_class(
    "input",
    param_names=["GroupSize", "ActiveInterval", "ActiveRate"],
    var_name_types=[("RefracTime", "scalar")],
    derived_params=[("TauRefrac", genn_model.create_dpf_class(lambda pars, dt: 1000.0 / pars[2])())],

    sim_code="""
    const unsigned int neuronGroup = $(id) / (unsigned int)$(GroupSize);
    const scalar groupStartTime = neuronGroup * $(ActiveInterval);
    const scalar groupEndTime = groupStartTime + $(ActiveInterval);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    """,
    threshold_condition_code="""
    $(t) > groupStartTime && $(t) < groupEndTime && $(RefracTime) <= 0.0
    """,
    is_auto_refractory_required=False)

output_model = genn_model.create_custom_neuron_class(
    "output",
    param_names=["TauOut", "Bias", "Freq1", "Freq2", "Freq3"],
    var_name_types=[("Y", "scalar"), ("YStar", "scalar"), ("E", "scalar"),
                    ("Ampl1", "scalar", VarAccess_READ_ONLY), ("Ampl2", "scalar", VarAccess_READ_ONLY), ("Ampl3", "scalar", VarAccess_READ_ONLY),
                    ("Phase1", "scalar", VarAccess_READ_ONLY), ("Phase2", "scalar", VarAccess_READ_ONLY), ("Phase3", "scalar", VarAccess_READ_ONLY)],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Freq1Radians", genn_model.create_dpf_class(lambda pars, dt: pars[2] * 2.0 * np.pi / 1000.0)()),
                    ("Freq2Radians", genn_model.create_dpf_class(lambda pars, dt: pars[3] * 2.0 * np.pi / 1000.0)()),
                    ("Freq3Radians", genn_model.create_dpf_class(lambda pars, dt: pars[4] * 2.0 * np.pi / 1000.0)())],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(Bias);
    $(YStar) = $(Ampl1) * sin(($(Freq1Radians) * $(t)) + $(Phase1));
    $(YStar) += $(Ampl2) * sin(($(Freq2Radians) * $(t)) + $(Phase2));
    $(YStar) += $(Ampl3) * sin(($(Freq3Radians) * $(t)) + $(Phase3));
    $(E) = $(Y) - $(YStar);
    """,

    is_auto_refractory_required=False)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (ADAM_BETA1 ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (ADAM_BETA2 ** adam_step))

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

def update_learning_rate(learning_rate, optimiser_custom_updates):
    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["eta"].view[:] = learning_rate

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Input population
input_params = {"GroupSize": 4, "ActiveInterval": 200.0, "ActiveRate": 100.0}
input_vars = {"RefracTime": 0.0}

# Recurrent population
recurrent_params = {"TauM": 20.0, "Vthresh": 0.61, "TauRefrac": 5.0}
recurrent_vars = {"V": 0.0, "RefracTime": 0.0, "E": 0.0}

# Output population
output_ampl_init = genn_model.init_var("Uniform", {"min": 0.5, "max": 2.0})
output_phase_init = genn_model.init_var("Uniform", {"min": 0.0, "max": 2.0 * np.pi})
output_params = {"TauOut": 20.0, "Bias": 0.0,
                 "Freq1": 2.0, "Freq2": 3.0, "Freq3": 5.0}
output_vars = {"Y": 0.0, "YStar": 0.0, "E": 0.0,
               "Ampl1": output_ampl_init, "Ampl2": output_ampl_init, "Ampl3": output_ampl_init,
               "Phase1": output_phase_init, "Phase2": output_phase_init, "Phase3": output_phase_init}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# eProp parameters common across all populations
eprop_lif_params = {"TauE": 20.0, "CReg": 3.0, "FTarget": 10.0,
                    "TauFAvg": 500.0}
eprop_deep_r_lif_params = {"TauE": 20.0, "CReg": 3.0, "FTarget": 10.0,
                           "TauFAvg": 500.0, "CL1": C_L1}
eprop_pre_vars = {"ZFilter": 0.0}
eprop_post_vars = {"Psi": 0.0, "FAvg": 0.0}

# Input->recurrent synapse parameters
input_recurrent_g = genn_model.init_var(eprop.absolute_normal_snippet if input_recurrent_deep_r else "Normal", 
                                        {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_INPUT)})
input_recurrent_vars = {"eFiltered": 0.0, "DeltaG": 0.0,
                        "g": input_recurrent_g}

# Recurrent->recurrent synapse parameters
recurrent_recurrent_g = genn_model.init_var(eprop.absolute_normal_snippet if recurrent_recurrent_deep_r else "Normal", 
                                            {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT)})
recurrent_recurrent_vars = {"eFiltered": 0.0, "DeltaG": 0.0,
                            "g": recurrent_recurrent_g}

# Recurrent->output synapse parameters
recurrent_output_params = {"TauE": 20.0}
recurrent_output_pre_vars = {"ZFilter": 0.0}
recurrent_output_vars = {"DeltaG": 0.0,
                         "g": genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT)})}

# Optimiser initialisation
adam_params = {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "epsilon": 1E-8}
adam_vars = {"m": 0.0, "v": 0.0}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "pattern_recognition")
model.dT = 1.0
model.timing_enabled = True
model._model.set_fuse_postsynaptic_models(True)
model._model.set_fuse_pre_post_weight_update_models(True)

# Add neuron populations
input = model.add_neuron_population("Input", NUM_INPUT, input_model,
                                    input_params, input_vars)

recurrent = model.add_neuron_population("RecurrentLIF", NUM_RECURRENT, eprop.recurrent_lif_model,
                                        recurrent_params, recurrent_vars)

output = model.add_neuron_population("Output", NUM_OUTPUT, output_model,
                                     output_params, output_vars)

# Turn on spike recording
input.spike_recording_enabled = True
recurrent.spike_recording_enabled = True

# Add synapse populations
input_recurrent_sparse_init = (genn_model.init_connectivity("FixedProbability", 
                                                            {"prob": args.input_recurrent_sparsity}) if sparse_input_recurrent
                               else None)
input_recurrent = model.add_synapse_population(
    "InputRecurrentLIF", "SPARSE_INDIVIDUALG" if sparse_input_recurrent else "DENSE_INDIVIDUALG", NO_DELAY,
    input, recurrent,
    eprop.eprop_lif_deep_r_model if input_recurrent_deep_r else eprop.eprop_lif_model, 
    eprop_deep_r_lif_params if input_recurrent_deep_r else eprop_lif_params,
    input_recurrent_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {},
    input_recurrent_sparse_init)

recurrent_output = model.add_synapse_population(
    "RecurrentLIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, output,
    eprop.output_learning_model, recurrent_output_params, recurrent_output_vars, recurrent_output_pre_vars, {},
    "DeltaCurr", {}, {})
output_recurrent = model.add_synapse_population(
    "OutputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
    output, recurrent,
    eprop.feedback_model, {}, {"g": 0.0}, {}, {},
    "DeltaCurr", {}, {})
output_recurrent.ps_target_var = "ISynFeedback"

recurrent_recurrent_sparse_init = (genn_model.init_connectivity("FixedProbability",
                                                                {"prob": args.recurrent_recurrent_sparsity}) if sparse_recurrent_recurrent
                                   else None)
recurrent_recurrent = model.add_synapse_population(
    "RecurrentLIFRecurrentLIF", "SPARSE_INDIVIDUALG" if sparse_recurrent_recurrent else "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, recurrent,
    eprop.eprop_lif_deep_r_model if recurrent_recurrent_deep_r else eprop.eprop_lif_model, 
    eprop_deep_r_lif_params if recurrent_recurrent_deep_r else eprop_lif_params,
    recurrent_recurrent_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {},
    recurrent_recurrent_sparse_init)

# Add custom update for calculating initial tranpose weights
model.add_custom_update("recurrent_hidden_transpose", "CalculateTranspose", "Transpose",
                        {}, {}, {"variable": genn_model.create_wu_var_ref(recurrent_output, "g", output_recurrent, "g")})

# Add custom updates for updating weights using Adam optimiser
input_recurrent_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent, "DeltaG"),
                                      "variable": genn_model.create_wu_var_ref(input_recurrent, "g")}
recurrent_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_output, "DeltaG"),
                                       "variable": genn_model.create_wu_var_ref(recurrent_output, "g", output_recurrent, "g")}
recurrent_recurrent_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_recurrent, "DeltaG"),
                                          "variable": genn_model.create_wu_var_ref(recurrent_recurrent, "g")}


#recurrent_output_optimiser = model.add_custom_update("recurrent_lif_output_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_model,
#                                                     adam_params, adam_vars, recurrent_output_optimiser_var_refs)
recurrent_output_optimiser = model.add_custom_update("recurrent_lif_output_optimiser", "GradientLearn", eprop.gradient_descent_zero_gradient_model,
                                                     {}, {}, recurrent_output_optimiser_var_refs)


# If we're using Deep-R on input-recurrent connectivity
if input_recurrent_deep_r:
    input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.gradient_descent_zero_gradient_track_dormant_model,
                                                        {}, {}, input_recurrent_optimiser_var_refs)
    #input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_sign_track_model,
    #                                                    adam_params, adam_vars, input_recurrent_optimiser_var_refs)
    input_recurrent_deep_r = DeepR(input_recurrent, input_recurrent_optimiser, 
                                   NUM_INPUT, NUM_RECURRENT, WEIGHT_0)
else:
    input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.gradient_descent_zero_gradient_model,
                                                        {}, {}, input_recurrent_optimiser_var_refs)
    #input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_model,
    #                                                    adam_params, adam_vars, input_recurrent_optimiser_var_refs)

# If we're using Deep-R on recurrent-recurrent connectivity
if recurrent_recurrent_deep_r:
    recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.gradient_descent_zero_gradient_track_dormant_model,
                                                            {}, {}, recurrent_recurrent_optimiser_var_refs)
    #recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_sign_track_model,
    #                                                        adam_params, adam_vars, recurrent_recurrent_optimiser_var_refs)
    recurrent_recurrent_deep_r = DeepR(recurrent_recurrent, recurrent_recurrent_optimiser, 
                                       NUM_RECURRENT, NUM_RECURRENT, WEIGHT_0)
else:
    recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.gradient_descent_zero_gradient_model,
                                                            {}, {}, recurrent_recurrent_optimiser_var_refs)
    #recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_model,
    #                                                        adam_params, adam_vars, recurrent_recurrent_optimiser_var_refs)


# Build model
model.build()

model.load(num_recording_timesteps=1000)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")

if input_recurrent_deep_r:
    input_recurrent_deep_r.load()
if recurrent_recurrent_deep_r:
    recurrent_recurrent_deep_r.load()
    
# Loop through trials
output_y_view = output.vars["Y"].view
output_y_star_view = output.vars["YStar"].view
learning_rate = 1E-5
#adam_step = 1
input_spikes = []
recurrent_spikes = []
output_y = []
output_y_star = []
deep_r_time = 0.0
for trial in range(1000):
    # Reduce learning rate every 100 trials
    if (trial % 200) == 0 and trial != 0:
        print(f"Trial {trial}")
        learning_rate *= 0.7
        
    record_trial = ((trial % 100) == 0)
    #record_trial = (trial == 999)

    # Reset time
    model.timestep = 0
    model.t = 0.0

    trial_output_y =[]
    trial_output_y_star = []
    for i in range(1000):
        model.step_time()

        if record_trial:
            output.pull_var_from_device("Y")
            output.pull_var_from_device("YStar")

            trial_output_y.append(np.copy(output_y_view))
            trial_output_y_star.append(np.copy(output_y_star_view))

    if record_trial:
        output_y.append(np.vstack(trial_output_y))
        output_y_star.append(np.vstack(trial_output_y_star))

        model.pull_recording_buffers_from_device()

        input_spikes.append(input.spike_recording_data)
        recurrent_spikes.append(recurrent.spike_recording_data)

    update_learning_rate(learning_rate, [input_recurrent_optimiser,
                                         recurrent_output_optimiser,
                                         recurrent_recurrent_optimiser])
    # Update Adam optimiser scaling factors
    #update_adam(learning_rate, adam_step, [input_recurrent_optimiser,
    #                                       recurrent_output_optimiser,
    #                                       recurrent_recurrent_optimiser])
    #adam_step += 1

    if args.deep_r:
        deep_r_reset_start = perf_counter()
        
        if sparse_input_recurrent:
            input_recurrent_deep_r.reset()
        if sparse_recurrent_recurrent:
            recurrent_recurrent_deep_r.reset()
        
        deep_r_reset_end = perf_counter()
        deep_r_time += (deep_r_reset_end - deep_r_reset_start)
    
    # Now batch is complete, apply gradients
    model.custom_update("GradientLearn")
    
    if args.deep_r:
        deep_r_update_start = perf_counter()
        
        if sparse_input_recurrent:
            input_recurrent_deep_r.update()
        if sparse_recurrent_recurrent:
            recurrent_recurrent_deep_r.update()
        
        deep_r_update_end = perf_counter()
        deep_r_time += (deep_r_update_end - deep_r_update_start)

print(f"Init: {model.init_time}")
print(f"Init sparse: {model.init_sparse_time}")
print(f"Neuron update: {model.neuron_update_time}")
print(f"Presynaptic update: {model.presynaptic_update_time}")
print(f"Synapse dynamics: {model.synapse_dynamics_time}")

if args.deep_r:
    print(f"Deep-R: {deep_r_time}")

assert len(input_spikes) == len(recurrent_spikes)
assert len(input_spikes) == len(output_y)
assert len(input_spikes) == len(output_y_star)

# Create plot
if args.plot:
    figure, axes = plt.subplots(5, len(output_y), sharex="col", sharey="row")

# Loop through recorded trials
for i, (s, r, y, y_star) in enumerate(zip(input_spikes, recurrent_spikes, output_y, output_y_star)):
    # Loop through output axes
    if args.plot:
        col_axes = axes if len(output_y) == 1 else axes[:, i]
    error = []
    for a in range(3):
        # Calculate error and hence MSE
        error.append(y[:,a] - y_star[:,a])
        mse = np.sum(error[-1] * error[-1]) / len(error[-1])
        #total_mse += mse
    
        if args.plot:
            # YA and YA*
            col_axes[a].plot(y[:,a])
            col_axes[a].plot(y_star[:,a])
            
            col_axes[a].set_title(f"Y{a} (MSE={mse:.2f})")
    error = np.hstack(error)
    total_mse = np.sum(error * error) / len(error)
    print(f"{i}: Total MSE: {total_mse}")
    
    if args.plot:
        # Input and recurrent spikes
        col_axes[3].scatter(s[0], s[1], s=1)
        col_axes[4].scatter(r[0], r[1], s=1)

if args.plot:
    col_axes = axes if len(output_y) == 1 else axes[:, 0]
    col_axes[0].set_ylim((-3.0, 3.0))
    col_axes[1].set_ylim((-3.0, 3.0))
    col_axes[2].set_ylim((-3.0, 3.0))
    plt.show()
