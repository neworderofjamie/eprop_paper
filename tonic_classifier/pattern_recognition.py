import numpy as np
import os
import matplotlib.pyplot as plt

from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

# Eprop imports
import eprop

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

WEIGHT_0 = 1.0

NUM_INPUT = 20
NUM_RECURRENT = 256
NUM_OUTPUT = 3

SPARSITY = 0.1
DEEP_R = True

class DeepR:
    def __init__(self, sg, optimiser, num_pre, num_post, weight_0):
        self.sg = sg
        self.optimiser = optimiser
        self.num_pre = num_pre
        self.num_post = num_post
        self.weight_sd = weight_0 / np.sqrt(self.num_pre)
        
        # Zero all bitmask EGPs
        num_words = int(np.ceil((num_pre * sg.max_row_length) / 32))
        optimiser.extra_global_params["signChange"].set_values(
            np.zeros(num_words, dtype=np.uint32))

    
    def load(self):
        # Download newly-generated connectivity
        self.sg.pull_connectivity_from_device()
    
        self.unpacked_conn = np.zeros((self.num_pre, self.num_post), dtype=np.uint8)
        self.unpacked_conn[self.sg.get_sparse_pre_inds(),
                           self.sg.get_sparse_post_inds()] = 1

        # Get views of sign change 
        self.sign_change_view = self.optimiser.extra_global_params["signChange"].view[:]

        # Get views to state variables
        self.sg_var_views = {n: v.view for n, v in self.sg.vars.items()}
        self.optimiser_var_views = {n: v.view for n, v in self.optimiser.vars.items()}
    
    def plot_sparse(self, axis):
        axis.set_title("Sparse")
        connectivity = np.zeros((self.num_pre, self.num_post))
        connectivity[self.sg.get_sparse_pre_inds(), 
                     self.sg.get_sparse_post_inds()] = 1
        axis.imshow(connectivity)
    
    def plot_unpacked(self, axis):
        axis.set_title("Unpacked")
        axis.imshow(self.unpacked_conn)
        
    def plot_sparse_unpacked_comparison(self, axis):
        axis.set_title("Comparison")
        connectivity = np.zeros((self.num_pre, self.num_post))
        connectivity[:] = self.unpacked_conn
        connectivity[self.sg.get_sparse_pre_inds(), 
                     self.sg.get_sparse_post_inds()] -= 1
        axis.imshow(connectivity)
    
    def reset(self):
        # Zero and upload sign change
        self.sign_change_view[:] = 0
        self.optimiser.push_extra_global_param_to_device("signChange")
   
    def update(self):
        # Download sign change
        self.optimiser.pull_extra_global_param_from_device("signChange")

        # Unpack sign change bitmasks
        sign_change_unpack = np.unpackbits(
            self.sign_change_view.view(dtype=np.uint8), 
            count=self.num_pre * self.sg.max_row_length, bitorder="little")
        
        # Reshape
        sign_change_unpack = np.reshape(sign_change_unpack, 
                                        (self.num_pre, self.sg.max_row_length))
        
        # Count dormant synapses
        total_dormant = np.sum(sign_change_unpack)

        # If no synapses have been made dormant, no need to do anything further
        if total_dormant == 0:
            return

        # Download optimiser and synapse group state
        self.optimiser.pull_state_from_device()
        self.sg.pull_state_from_device()

        # Loop through rows
        for i in range(self.num_pre):
            row_length = self.sg._row_lengths[i]
            start_id = self.sg.max_row_length * i
            end_id = start_id + row_length
            inds = self.sg._ind[start_id:end_id]

            # Select inds in this row which will be made dormant
            dormant_inds = inds[sign_change_unpack[i,:row_length] == 1]

            # If there are any
            num_dormant = len(dormant_inds)
            if num_dormant > 0:
                # Check there is enough row left to make this many synapses dormant
                assert row_length >= num_dormant
                
                # Get mask of row entries to keep
                keep_mask = (sign_change_unpack[i,:row_length] == 0)
                slice_length = np.sum(keep_mask)
                assert slice_length == (row_length - num_dormant)
            
                # Clear dormant synapses from unpacked representation
                self.unpacked_conn[i, dormant_inds] = 0
                
                # Remove inactive indices
                self.sg._ind[start_id:start_id + slice_length] = inds[keep_mask]
           
                # Remove inactive synapse group state vars
                for v in self.sg_var_views.values():
                    v_row = v[start_id:end_id]
                    v[start_id:start_id + slice_length] = v_row[keep_mask]
                
                # Remove inactive optimiser state vars
                for v in self.optimiser_var_views.values():
                    v_row = v[start_id:end_id]
                    v[start_id:start_id + slice_length] = v_row[keep_mask]
                    
                # Reduce row length
                self.sg._row_lengths[i] -= num_dormant

        # Count number of remaining synapses
        num_synapses = np.sum(self.sg._row_lengths)

        # From this, calculate how many padding synapses there are in data structure
        num_total_padding_synapses = (self.sg.max_row_length * self.num_pre) - num_synapses

        # Loop through rows of synaptic matrix
        num_activations = np.zeros(self.num_pre, dtype=int)
        for i in range(self.num_pre - 1):
            num_row_padding_synapses = self.sg.max_row_length - self.sg._row_lengths[i]
            
            if num_row_padding_synapses > 0 and num_total_padding_synapses > 0:
                probability = num_row_padding_synapses / num_total_padding_synapses
                
                # Sample number of activations
                num_row_activations = min(num_row_padding_synapses, 
                                          np.random.binomial(total_dormant, probability))
                num_activations[i] = num_row_activations;

                # Update counters
                total_dormant -= num_row_activations
                num_total_padding_synapses -= num_row_padding_synapses

        # Put remainder of activations in last row
        assert total_dormant < (self.sg.max_row_length - self.sg._row_lengths[-1])
        num_activations[-1] = total_dormant;

        # Loop through rows
        for i in range(self.num_pre):
            # If there's anything to activate on this row
            if num_activations[i] > 0:
                new_syn_start_ind = (self.sg.max_row_length * i) + self.sg._row_lengths[i]
                new_syn_end_ind = new_syn_start_ind + num_activations[i]
                
                # Get possible inds to chose from
                possible_inds = np.where(self.unpacked_conn[i] == 0)
                
                new_inds = np.random.choice(possible_inds[0], num_activations[i],
                                            replace=False)
                # Sample indices
                self.sg._ind[new_syn_start_ind:new_syn_end_ind] = new_inds
                
                # Update connectivity bitmask
                self.unpacked_conn[i, new_inds] = 1

                # Initialise synapse group state variables
                for n, v in self.sg_var_views.items():
                    if n == "g":
                        v[new_syn_start_ind:new_syn_end_ind] =\
                            np.random.normal(0.0, self.weight_sd, num_activations[i])
                    else:
                        v[new_syn_start_ind:new_syn_end_ind] = 0
                
                # Initialise optimiser state variables
                for v in self.optimiser_var_views.values():
                    v[new_syn_start_ind:new_syn_end_ind] = 0
            
                # Update row length
                self.sg._row_lengths[i] += num_activations[i]

        # Upload optimiser and synapse group state
        self.optimiser.push_state_to_device()
        self.sg.push_state_to_device()
        self.sg.push_connectivity_to_device()
        
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
eprop_pre_vars = {"ZFilter": 0.0}
eprop_post_vars = {"Psi": 0.0, "FAvg": 0.0}

# Input->recurrent synapse parameters
input_recurrent_vars = {"eFiltered": 0.0, "DeltaG": 0.0,
                        "g": genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_INPUT)})}

# Recurrent->recurrent synapse parameters
recurrent_recurrent_vars = {"eFiltered": 0.0, "DeltaG": 0.0,
                            "g": genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT)})}

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
input_recurrent_sparse_init = (None if SPARSITY is None 
                               else genn_model.init_connectivity("FixedProbability", {"prob": SPARSITY}))
input_recurrent = model.add_synapse_population(
    "InputRecurrentLIF", "DENSE_INDIVIDUALG" if SPARSITY is None else "SPARSE_INDIVIDUALG", NO_DELAY,
    input, recurrent,
    eprop.eprop_lif_model, eprop_lif_params, input_recurrent_vars, eprop_pre_vars, eprop_post_vars,
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

recurrent_recurrent_sparse_init = (None if SPARSITY is None 
                                   else genn_model.init_connectivity("FixedProbabilityNoAutapse", {"prob": SPARSITY}))
recurrent_recurrent = model.add_synapse_population(
    "RecurrentLIFRecurrentLIF", "DENSE_INDIVIDUALG" if SPARSITY is None else "SPARSE_INDIVIDUALG", NO_DELAY,
    recurrent, recurrent,
    eprop.eprop_lif_model, eprop_lif_params, recurrent_recurrent_vars, eprop_pre_vars, eprop_post_vars,
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


recurrent_output_optimiser = model.add_custom_update("recurrent_lif_output_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_model,
                                                     adam_params, adam_vars, recurrent_output_optimiser_var_refs)

# If we're using Deep-R
if DEEP_R:
    input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_sign_track_model,
                                                        adam_params, adam_vars, input_recurrent_optimiser_var_refs)
    recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_sign_track_model,
                                                            adam_params, adam_vars, recurrent_recurrent_optimiser_var_refs)
    
    input_recurrent_deep_r = DeepR(input_recurrent, input_recurrent_optimiser, 
                                   NUM_INPUT, NUM_RECURRENT, WEIGHT_0)
    recurrent_recurrent_deep_r = DeepR(recurrent_recurrent, recurrent_recurrent_optimiser, 
                                       NUM_RECURRENT, NUM_RECURRENT, WEIGHT_0)
else:
    input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_model,
                                                        adam_params, adam_vars, input_recurrent_optimiser_var_refs)
    recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_zero_gradient_model,
                                                            adam_params, adam_vars, recurrent_recurrent_optimiser_var_refs)


# Build model
model.build()


model.load(num_recording_timesteps=1000)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")

if DEEP_R:
    input_recurrent_deep_r.load()
    recurrent_recurrent_deep_r.load()
    
# Loop through trials
output_y_view = output.vars["Y"].view
output_y_star_view = output.vars["YStar"].view
learning_rate = 0.003
adam_step = 1
input_spikes = []
recurrent_spikes = []
output_y = []
output_y_star = []
deep_r_update_time = 0.0
deep_r_reset_time = 0.0
for trial in range(1000):
    # Reduce learning rate every 100 trials
    if (trial % 100) == 0 and trial != 0:
        print(f"Trial {trial}")
        learning_rate *= 0.7

    record_trial = ((trial % 100) == 0)

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

    # Update Adam optimiser scaling factors
    update_adam(learning_rate, adam_step, [input_recurrent_optimiser,
                                           recurrent_output_optimiser,
                                           recurrent_recurrent_optimiser])
    adam_step += 1

    if DEEP_R:
        deep_r_reset_start = perf_counter()
        
        input_recurrent_deep_r.reset()
        recurrent_recurrent_deep_r.reset()
        
        deep_r_reset_end = perf_counter()
        deep_r_reset_time += (deep_r_reset_end - deep_r_reset_start)
    
    # Now batch is complete, apply gradients
    model.custom_update("GradientLearn")
    
    if DEEP_R:
        deep_r_update_start = perf_counter()
        
        input_recurrent_deep_r.update()
        recurrent_recurrent_deep_r.update()
        
        deep_r_update_end = perf_counter()
        deep_r_update_time += (deep_r_update_end - deep_r_update_start)

print(f"Init: {model.init_time}")
print(f"Init sparse: {model.init_sparse_time}")
print(f"Neuron update: {model.neuron_update_time}")
print(f"Presynaptic update: {model.presynaptic_update_time}")
print(f"Synapse dynamics: {model.synapse_dynamics_time}")

if DEEP_R:
    print(f"Deep-R reset: {deep_r_reset_time}")
    print(f"Deep-R update: {deep_r_update_time}")

assert len(input_spikes) == len(recurrent_spikes)
assert len(input_spikes) == len(output_y)
assert len(input_spikes) == len(output_y_star)

# Create plot
figure, axes = plt.subplots(5, len(output_y), sharex="col", sharey="row")

# Loop through recorded trials
for i, (s, r, y, y_star) in enumerate(zip(input_spikes, recurrent_spikes, output_y, output_y_star)):
    # Loop through output axes
    total_mse = 0.0
    for a in range(3):
        # Calculate error and hence MSE
        error = y[:,a] - y_star[:,a]
        mse = np.sum(error * error) / len(error)
        
        # YA and YA*
        axes[a, i].plot(y[:,a])
        axes[a, i].plot(y_star[:,a])
        
        axes[a, i].set_title(f"Y{a} (MSE={mse:.2f})")
        total_mse += mse
    print(f"{i}: Total MSE: {total_mse}")
    
    # Input and recurrent spikes
    axes[3, i].scatter(s[0], s[1], s=1)
    axes[4, i].scatter(r[0], r[1], s=1)

axes[0, 0].set_ylim((-3.0, 3.0))
axes[1, 0].set_ylim((-3.0, 3.0))
axes[2, 0].set_ylim((-3.0, 3.0))
plt.show()
