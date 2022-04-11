import numpy as np
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

# Eprop imports
import eprop

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

WEIGHT_0 = 1.0

NUM_INPUT = 20
NUM_RECURRENT_LIF = 600
NUM_OUTPUT = 3

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
input_model = genn_model.create_custom_neuron_class(
    "input",
    param_names=["GroupSize", "ActiveInterval",
                 "ActiveRate", "PatternLength"],
    var_name_types=[("RefracTime", "scalar")],
    derived_params=[("TauRefrac", genn_model.create_dpf_class(lambda pars, dt: 1000.0 / pars[2])())],

    sim_code="""
    const scalar tPattern = fmod($(t), $(PatternLength));
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
    tPattern > groupStartTime && tPattern < groupEndTime && $(RefracTime) <= 0.0
    """,
    is_auto_refractory_required=False)

output_model = genn_model.create_custom_neuron_class(
    "output",
    param_names=["TauOut", "Bias", "Freq1", "Freq2", "Freq3", "PatternLength"],
    var_name_types=[("Y", "scalar"), ("YStar", "scalar"), ("E", "scalar"),
                    ("Ampl1", "scalar", VarAccess_READ_ONLY), ("Ampl2", "scalar", VarAccess_READ_ONLY), ("Ampl3", "scalar", VarAccess_READ_ONLY),
                    ("Phase1", "scalar", VarAccess_READ_ONLY), ("Phase2", "scalar", VarAccess_READ_ONLY), ("Phase3", "scalar", VarAccess_READ_ONLY)],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Freq1Radians", genn_model.create_dpf_class(lambda pars, dt: pars[2] * 2.0 * np.pi / 1000.0)()),
                    ("Freq2Radians", genn_model.create_dpf_class(lambda pars, dt: pars[3] * 2.0 * np.pi / 1000.0)()),
                    ("Freq3Radians", genn_model.create_dpf_class(lambda pars, dt: pars[4] * 2.0 * np.pi / 1000.0)())],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(Bias);
    const scalar tPattern = fmod($(t), $(PatternLength));
    $(YStar) = $(Ampl1) * sin(($(Freq1Radians) * tPattern) + $(Phase1));
    $(YStar) += $(Ampl2) * sin(($(Freq2Radians) * tPattern) + $(Phase2));
    $(YStar) += $(Ampl3) * sin(($(Freq3Radians) * tPattern) + $(Phase3));
    $(E) = $(Y) - $(YStar);
    """,

    is_auto_refractory_required=False)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def write_spike_file(filename, data):
    np.savetxt(filename, np.column_stack(data), fmt=["%f","%d"], 
               delimiter=",", header="Time [ms], Neuron ID")

def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (ADAM_BETA1 ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (ADAM_BETA2 ** adam_step))

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Input population
input_params = {"GroupSize": 4, "ActiveInterval": 200.0,
                "ActiveRate": 100.0, "PatternLength": 1000.0}
input_vars = {"RefracTime": 0.0}

# Recurrent population
recurrent_lif_params = {"TauM": 20.0, "Vthresh": 0.6, "TauRefrac": 5.0}
recurrent_lif_vars = {"V": 0.0, "RefracTime": 0.0, "E": 0.0}

# Output population
output_ampl_init = genn_model.init_var("Uniform", {"min": 0.5, "max": 2.0})
output_phase_init = genn_model.init_var("Uniform", {"min": 0.0, "max": 2.0 * np.pi})
output_params = {"TauOut": 20.0, "Bias": 0.0, 
                 "Freq1": 2.0, "Freq2": 3.0, "Freq3": 5.0, 
                 "PatternLength": 1000.0}
output_vars = {"Y": 0.0, "YStar": 0.0, "E": 0.0, 
               "Ampl1": output_ampl_init, "Ampl2": output_ampl_init, "Ampl3": output_ampl_init, 
               "Phase1": output_phase_init, "Phase2": output_phase_init, "Phase3": output_phase_init}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# eProp parameters common across all populations
eprop_lif_params = {"TauE": 20.0, "CReg": 3.0, "FTarget": 10.0, 
                    "TauFAvg": 500.0}
eprop_pre_vars = {"ZFilter": 0.0}
eprop_post_vars = {"Psi": 0.0, "FAvg": 0.0}

# Input->recurrent synapse parameters
input_recurrent_lif_vars = {"eFiltered": 0.0, "DeltaG": 0.0,
                            "g": genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_INPUT)})}

# Recurrent->recurrent synapse parameters
recurrent_lif_recurrent_lif_vars = {"eFiltered": 0.0, "DeltaG": 0.0, 
                                    "g": genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT_LIF)})}

# Recurrent->output synapse parameters
recurrent_output_params = {"TauE": 20.0}
recurrent_output_pre_vars = {"ZFilter": 0.0}
recurrent_lif_output_vars = {"DeltaG": 0.0,
                             "g": genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT_LIF)})}

# Optimiser initialisation
adam_params = {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "epsilon": 1E-8}
adam_vars = {"m": 0.0, "v": 0.0}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "pattern_recognition")
model.dT = 1.0
#model.timing_enabled = 
model._model.set_fuse_postsynaptic_models(True)
model._model.set_fuse_pre_post_weight_update_models(True)

# Add neuron populations
input = model.add_neuron_population("Input", NUM_INPUT, input_model,
                                    input_params, input_vars)

recurrent_lif = model.add_neuron_population("RecurrentLIF", NUM_RECURRENT_LIF, eprop.recurrent_lif_model,
                                            recurrent_lif_params, recurrent_lif_vars)

output = model.add_neuron_population("Output", NUM_OUTPUT, output_model,
                                     output_params, output_vars)

# Add synapse populations
input_recurrent_lif = model.add_synapse_population(
    "InputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
    input, recurrent_lif,
    eprop.eprop_lif_model, eprop_lif_params, input_recurrent_lif_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {})
recurrent_lif_output = model.add_synapse_population(
    "RecurrentLIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent_lif, output,
    eprop.output_learning_model, recurrent_output_params, recurrent_lif_output_vars, recurrent_output_pre_vars, {},
    "DeltaCurr", {}, {})
output_recurrent_lif = model.add_synapse_population(
    "OutputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
    output, recurrent_lif,
    eprop.feedback_model, {}, {"g": 0.0}, {}, {},
    "DeltaCurr", {}, {})
output_recurrent_lif.ps_target_var = "ISynFeedback"

recurrent_lif_recurrent_lif = model.add_synapse_population(
    "RecurrentLIFRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent_lif, recurrent_lif,
    eprop.eprop_lif_model, eprop_lif_params, recurrent_lif_recurrent_lif_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {})

# Add custom update for calculating initial tranpose weights
model.add_custom_update("recurrent_lif_hidden_transpose", "CalculateTranspose", "Transpose",
                        {}, {}, {"variable": genn_model.create_wu_var_ref(recurrent_lif_output, "g", output_recurrent_lif, "g")})

# Add custom updates for updating reduced weights using Adam optimiser
input_recurrent_lif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_lif, "DeltaG"),
                                          "variable": genn_model.create_wu_var_ref(input_recurrent_lif, "g")}
input_recurrent_lif_optimiser = model.add_custom_update("input_recurrent_lif_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                        adam_params, adam_vars, input_recurrent_lif_optimiser_var_refs)
recurrent_lif_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_output, "DeltaG"),
                                           "variable": genn_model.create_wu_var_ref(recurrent_lif_output, "g" , output_recurrent_lif, "g")}
recurrent_lif_output_optimiser = model.add_custom_update("recurrent_lif_output_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                         adam_params, adam_vars, recurrent_lif_output_optimiser_var_refs)

recurrent_lif_recurrent_lif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif, "DeltaG"),
                                                  "variable": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif, "g")}
recurrent_lif_recurrent_lif_optimiser = model.add_custom_update("recurrent_lif_recurrent_alif_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                                adam_params, adam_vars, recurrent_lif_recurrent_lif_optimiser_var_refs)

# Build model
model.build()


model.load(num_recording_timesteps=1000)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")