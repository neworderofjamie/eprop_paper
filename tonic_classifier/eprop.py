import numpy as np

from pygenn import (create_custom_update_model, create_neuron_model,
                    create_var_init_snippet, create_weight_update_model)
from pygenn import CustomUpdateVarAccess, VarAccess, VarAccessMode

# ----------------------------------------------------------------------------
# Var init snippets
# ----------------------------------------------------------------------------
absolute_normal_snippet = create_var_init_snippet(
    "absolute_normal",
    params=["mean", "sd"],
    var_init_code="""
    value = fabs(mean + (gennrand_normal * sd));
    """)

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
adam_optimizer_model = create_custom_update_model(
    "adam_optimizer",
    params=["beta1", "beta2", "epsilon", "alpha", 
           "firstMomentScale", "secondMomentScale"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    var_refs=[("gradient", "scalar", VarAccessMode.READ_ONLY), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    m = (beta1 * m) + ((1.0 - beta1) * gradient);

    // Update biased second moment estimate
    v = (beta2 * v) + ((1.0 - beta2) * gradient * gradient);

    // Add gradient to variable, scaled by learning rate
    variable -= (alpha * m * firstMomentScale) / (sqrt(v * secondMomentScale) + epsilon);
    """)

adam_optimizer_track_dormant_model = create_custom_update_model(
    "adam_optimizer_track_dormant",
    params=["beta1", "beta2", "epsilon", "alpha", 
            "firstMomentScale", "secondMomentScale"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    extra_global_params=[("dormant", "uint32_t*")],
    var_refs=[("gradient", "scalar", VarAccessMode.READ_ONLY), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    m = (beta1 * m) + ((1.0 - beta1) * gradient);

    // Update biased second moment estimate
    v = (beta2 * v) + ((1.0 - beta2) * gradient * gradient);

    // Add gradient to variable, scaled by learning rate
    variable -= (alpha * m * firstMomentScale) / (sqrt(v * secondMomentScale) + epsilon);

    // If variable has now gone negative, mark as dormant
    if(variable < 0.0) {
        atomicOr(&dormant[id_syn / 32], 1 << (id_syn % 32));
    }
    """)


adam_optimizer_zero_gradient_model = create_custom_update_model(
    "adam_optimizer_zero_gradient",
    params=["beta1", "beta2", "epsilon", "alpha", 
            "firstMomentScale", "secondMomentScale"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    var_refs=[("gradient", "scalar"), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    m = (beta1 * m) + ((1.0 - beta1) * gradient);

    // Update biased second moment estimate
    v = (beta2 * v) + ((1.0 - beta2) * gradient * gradient);

    // Add gradient to variable, scaled by learning rate
    variable -= (alpha * m * firstMomentScale) / (sqrt(v * secondMomentScale) + epsilon);

    // Zero gradient
    gradient = 0.0;
    """)

adam_optimizer_zero_gradient_track_dormant_model = create_custom_update_model(
    "adam_optimizer_zero_gradient_track_dormant",
    params=["beta1", "beta2", "epsilon", "alpha", 
            "firstMomentScale", "secondMomentScale"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    extra_global_params=[("dormant", "uint32_t*")],
    var_refs=[("gradient", "scalar"), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    m = (beta1 * m) + ((1.0 - beta1) * gradient);

    // Update biased second moment estimate
    v = (beta2 * v) + ((1.0 - beta2) * gradient * gradient);

    // Add gradient to variable, scaled by learning rate
    variable -= (alpha * m * firstMomentScale) / (sqrt(v * secondMomentScale) + epsilon);

    // Zero gradient
    gradient = 0.0;

    // If variable started has gone negative, mark as dormant
    if(variable < 0.0) {
        atomicOr(&dormant[id_syn / 32], 1 << (id_syn % 32));
    }
    """)

l1_model = create_custom_update_model(
    "l1",
    params=["c"],
    var_refs=[("variable", "scalar")],
    update_code="""
    variable += c;
    """)
    
gradient_descent_zero_gradient_model = create_custom_update_model(
    "gradient_descent_zero_gradient",
    params=["eta"],
    var_refs=[("gradient", "scalar"), ("variable", "scalar")],
    update_code="""
    // Descend!
    variable -= eta * gradient;

    // Zero gradient
    gradient = 0.0;
    """)

gradient_descent_zero_gradient_track_dormant_model = create_custom_update_model(
    "gradient_descent_zero_gradient",
    extra_global_params=[("eta", "scalar"), ("dormant", "uint32_t*")],
    var_refs=[("gradient", "scalar"), ("variable", "scalar")],
    update_code="""
    // Descend!
    variable -= eta * gradient;

    // Zero gradient
    gradient = 0.0;

    // If variable has gone negative, mark as dormant
    if(variable < 0.0) {
        atomicOr(&dormant[id_syn / 32], 1 << (id_syn % 32));
    }
    """)

gradient_batch_reduce_model = create_custom_update_model(
    "gradient_batch_reduce",
    var_name_types=[("reducedGradient", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
    var_refs=[("gradient", "scalar")],
    update_code="""
    reducedGradient = gradient;
    gradient = 0;
    """)

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
recurrent_alif_model = create_neuron_model(
    "recurrent_alif",
    params=["TauM", "TauAdap", "Vthresh", "TauRefrac", "Beta"],
    var_name_types=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauM"])),
                    ("Rho", lambda pars, dt: np.exp(-dt / pars["TauAdap"]))],

    sim_code="""
    E = ISynFeedback;
    V = (Alpha * V) + Isyn;
    A *= Rho;
    if (RefracTime > 0.0) {
      RefracTime -= dt;
    }
    """,
    reset_code="""
    RefracTime = TauRefrac;
    V -= Vthresh;
    A += 1.0;
    """,
    threshold_condition_code="""
    RefracTime <= 0.0 && V >= (Vthresh + (Beta * A))
    """,
    is_auto_refractory_required=False)

recurrent_lif_model = create_neuron_model(
    "recurrent_lif",
    params=["TauM", "Vthresh", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauM"]))],
   
    sim_code="""
    E = ISynFeedback;
    V = (Alpha * V) + Isyn;
    if (RefracTime > 0.0) {
      RefracTime -= dt;
    }
    """,
    reset_code="""
    RefracTime = TauRefrac;
    V -= Vthresh;
    """,
    threshold_condition_code="""
    RefracTime <= 0.0 && V >= Vthresh
    """,
    is_auto_refractory_required=False)

#----------------------------------------------------------------------------
# Weight update models
#----------------------------------------------------------------------------
eprop_alif_model = create_weight_update_model(
    "eprop_alif",
    params=["TauE", "TauA", "CReg", "FTarget", "TauFAvg", "Beta", "Vthresh"],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauE"])),
                    ("Rho", lambda pars, dt: np.exp(-dt / pars["TauA"])),
                    ("FTargetTimestep", lambda pars, dt: (pars["FTarget"] * dt) / 1000.0),
                    ("AlphaFAv", lambda pars, dt: np.exp(-dt / pars["TauFAvg"]))],
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY), ("eFiltered", "scalar"), ("epsilonA", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    post_neuron_var_refs=[("RefracTime_post", "scalar"), ("V_post", "scalar"), ("A_post", "scalar"), ("E_post", "scalar")],

    sim_code="""
    addToPost(g);
    """,

    pre_spike_code="""
    ZFilter += 1.0;
    """,
    pre_dynamics_code="""
    ZFilter *= Alpha;
    """,

    post_spike_code="""
    FAvg += (1.0 - AlphaFAv);
    """,
    post_dynamics_code="""
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh + (Beta * A_post))) / Vthresh));
    }
    """,

    synapse_dynamics_code="""
    // Calculate some common factors in e and epsilon update
    scalar epsilonA = epsilonA;
    const scalar psiZFilter = Psi * ZFilter;
    const scalar psiBetaEpsilonA = Psi * Beta * epsilonA;
    
    // Calculate e and episilonA
    const scalar e = psiZFilter  - psiBetaEpsilonA;
    epsilonA = psiZFilter + ((Rho * epsilonA) - psiBetaEpsilonA);
    
    // Calculate filtered version of eligibility trace
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;
    
    // Apply weight update
    DeltaG += (eF * E_post) + ((FAvg - FTargetTimestep) * CReg * e);
    eFiltered = eF;
    """)
    
eprop_alif_deep_r_model = create_weight_update_model(
    "eprop_alif_deep_r",
    params=["TauE", "TauA", "CReg", "FTarget", "TauFAvg", "Beta", "Vthresh", ("NumExcitatory", "int")],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauE"])),
                    ("Rho", lambda pars, dt: np.exp(-dt / pars["TauA"])),
                    ("FTargetTimestep", lambda pars, dt: (pars["FTarget"] * dt) / 1000.0),
                    ("AlphaFAv", lambda pars, dt: np.exp(-dt / pars["TauFAvg"]))],
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY), ("eFiltered", "scalar"), ("epsilonA", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    post_neuron_var_refs=[("RefracTime_post", "scalar"), ("V_post", "scalar"), ("A_post", "scalar"), ("E_post", "scalar")],

    sim_code="""
    const float sign = (id_pre < NumExcitatory) ? 1.0 : -1.0;
    addToPost(sign * g);
    """,

    pre_spike_code="""
    ZFilter += 1.0;
    """,
    pre_dynamics_code="""
    ZFilter *= Alpha;
    """,

    post_spike_code="""
    FAvg += (1.0 - AlphaFAv);
    """,
    post_dynamics_code="""
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh + (Beta * A_post))) / Vthresh));
    }
    """,

    synapse_dynamics_code="""
    const float sign = (id_pre < NumExcitatory) ? 1.0 : -1.0;

    // Calculate some common factors in e and epsilon update
    scalar epsilonA = epsilonA;
    const scalar psiZFilter = Psi * ZFilter;
    const scalar psiBetaEpsilonA = Psi * Beta * epsilonA;

    // Calculate e and episilonA
    const scalar e = psiZFilter  - psiBetaEpsilonA;
    epsilonA = psiZFilter + ((Rho * epsilonA) - psiBetaEpsilonA);

    // Calculate filtered version of eligibility trace
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;

    // Apply weight update
    DeltaG += sign * ((eF * E_post) + ((FAvg - FTargetTimestep) * CReg * e));
    eFiltered = eF;
    """)

eprop_lif_model = create_weight_update_model(
    "eprop_lif",
    params=["TauE", "CReg", "FTarget", "TauFAvg", "Vthresh"],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauE"])),
                    ("FTargetTimestep", lambda pars, dt: (pars["FTarget"] * dt) / 1000.0),
                    ("AlphaFAv", lambda pars, dt: np.exp(-dt / pars["TauFAvg"]))],
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY), ("eFiltered", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    post_neuron_var_refs=[("RefracTime_post", "scalar"), ("V_post", "scalar"), ("E_post", "scalar")],

    sim_code="""
    addToPost(g);
    """,

    pre_spike_code="""
    ZFilter += 1.0;
    """,
    pre_dynamics_code="""
    ZFilter *= Alpha;
    """,

    post_spike_code="""
    FAvg += (1.0 - AlphaFAv);
    """,
    post_dynamics_code="""
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - Vthresh) / Vthresh));
    }
    """,

    synapse_dynamics_code="""
    const scalar e = ZFilter * Psi;
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;
    DeltaG += (eF * E_post) + ((FAvg - FTargetTimestep) * CReg * e);
    eFiltered = eF;
    """)

eprop_lif_deep_r_model = create_weight_update_model(
    "eprop_lif_deep_r",
    params=["TauE", "CReg", "FTarget", "TauFAvg", "Vthresh", ("NumExcitatory", "int")],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauE"])),
                    ("FTargetTimestep", lambda pars, dt: (pars["FTarget"] * dt) / 1000.0),
                    ("AlphaFAv", lambda pars, dt: np.exp(-dt / pars["TauFAvg"]))],
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY), ("eFiltered", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    post_neuron_var_refs=[("RefracTime_post", "scalar"), ("V_post", "scalar"), ("E_post", "scalar")],
    
    sim_code="""
    const float sign = (id_pre < NumExcitatory) ? 1.0 : -1.0;
    addToPost(sign * g);
    """,

    pre_spike_code="""
    ZFilter += 1.0;
    """,
    pre_dynamics_code="""
    ZFilter *= Alpha;
    """,

    post_spike_code="""
    FAvg += (1.0 - AlphaFAv);
    """,
    post_dynamics_code="""
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - Vthresh) / Vthresh));
    }
    """,

    synapse_dynamics_code="""
    // **HACK** get sign
    const float sign = (id_pre < NumExcitatory) ? 1.0 : -1.0;

    const scalar e = ZFilter * Psi;
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;
    DeltaG += sign * ((eFiltered * E_post) + ((FAvg - FTargetTimestep) * CReg * e));
    eFiltered = eF;
    """)
    
output_learning_model = create_weight_update_model(
    "output_learning",
    params=["TauE"],
    derived_params=[("Alpha", lambda pars, dt: np.exp(-dt / pars["TauE"]))],
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_neuron_var_refs=[("E_post", "scalar")],

    sim_code="""
    addToPost(g);
    """,

    pre_spike_code="""
    ZFilter += 1.0;
    """,
    pre_dynamics_code="""
    ZFilter *= Alpha;
    """,

    synapse_dynamics_code="""
    DeltaG += ZFilter * E_post;
    addToPre(g * E_post);
    """)
