import numpy as np

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