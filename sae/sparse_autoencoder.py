import gzip
import os
import pickle
import einops

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from tqdm.auto import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint

from sae.utils import log, get_gpu_memory


class SparseAutoencoder(HookedRootModule):
    """
    Sparse Autoencoder object. Takes a Config object as input
    """

    def __init__(self, cfg, init_weights=True):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = int(cfg.d_in * cfg.expansion_factor)
        self.l1_coefficient = cfg.l1_coefficient
        self.dtype = cfg.dtype
        self.device = torch.device(cfg.device)
        self.input_scaling_factor = None
        
        if self.cfg.different_output:
            self.d_out = self.cfg.d_out
        else:
            self.d_out = self.d_in

        if init_weights:
            self.set_initial_weights()
            self.customise_initial_weights()

        # Needed for HookedRootModule
        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()
        self.setup()

    def set_initial_weights(self):
        """
        Initialise encoder and decoder weights
        """
        # Initialise encoder
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
            torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae, dtype=self.dtype, device=self.device))

        # Initialise decoder
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
            torch.empty(self.d_sae, self.d_out, dtype=self.dtype, device=self.device))
        )
        self.b_dec = nn.Parameter(torch.zeros(self.d_out, dtype=self.dtype, device=self.device))

        if self.cfg.subtract_b_dec_from_inputs and self.cfg.different_output:
            self.b_pre = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))
        else:
            self.b_pre = self.b_dec

    @torch.no_grad()
    def customise_initial_weights(self):
        """
        Customise initial encoder and decoder based on config
        """
        # Random decoder directions with given norm
        if self.cfg.normalise_initial_decoder_weights:
            self.W_dec = nn.Parameter(torch.rand(self.d_sae, self.d_in, dtype=self.dtype, device=self.device))

            decoder_norms = torch.linalg.norm(self.W_dec.data, dim=1, keepdim=True)
            self.W_dec.data = self.cfg.initial_decoder_norm * self.W_dec.data / decoder_norms

        # Set W_enc to W_dec.T
        if self.cfg.initialise_encoder_to_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()

        # # Set decoder to mean activation
        # if self.cfg.b_dec_init_method == "mean":
        #     self.initialize_b_dec_with_mean(self.activation_store)

        # # Calculate scaling factor
        # if self.cfg.scale_input_norm:
        #     self.calculate_constant_input_scaling_factor(self.activation_store)

    def forward(self, input_activations):
        """
        Computes the forward pass
        """
        # Scale inputs if needed:
        if self.input_scaling_factor is not None:
            input_activations = self.input_scaling_factor * input_activations

        # Subtract decoder bias if desired
        if self.cfg.subtract_b_dec_from_inputs:
            input_activations = self.hook_sae_in(input_activations - self.b_pre)  

        self.hidden_pre = self.hook_hidden_pre(
            einops.einsum(input_activations, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")
            + self.b_enc
        )

        if self.cfg.activation_function == "relu":
            self.feature_activations = self.hook_hidden_post(torch.nn.functional.relu(self.hidden_pre))
        elif self.cfg.activation_function == "topk":
            topk = torch.topk(self.hidden_pre, k=self.cfg.topk_amount, dim=-1)
            values = torch.nn.functional.relu(topk.values)
            # make all other values 0
            self.feature_activations = torch.zeros_like(self.hidden_pre)
            self.feature_activations.scatter_(-1, topk.indices, values)
            self.feature_activations = self.hook_hidden_post(self.feature_activations)
            

        self.output_activations = self.hook_sae_out(
            einops.einsum(self.feature_activations, self.W_dec, "... d_sae, d_sae d_in -> ... d_in")
            + self.b_dec
        )
        
        # Unscale outputs if needed:
        if self.input_scaling_factor is not None:
            self.output_activations = self.output_activations / self.input_scaling_factor

        return self.feature_activations, self.output_activations

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, activation_store):
        """
        Subtract decoder bias from input activations
        """
        if self.cfg.different_output:
            all_activations = activation_store.storage_buffers[0].detach().cpu()
        else:
            all_activations = activation_store.storage_buffer.detach().cpu()
            
        out = all_activations.mean(dim=0)

        self.b_pre.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """
        Normalise decoder
        """
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, path: str):
        """
        Saves state_dict and config
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}
        # print("saving", self.cfg)

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        log(f"Saved model to {path}")
        log(get_gpu_memory())

    @classmethod
    def load_from_pretrained(cls, path: str):
        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        try:
            if torch.backends.mps.is_available():
                state_dict = torch.load(path, map_location="mps")
                state_dict["cfg"].device = "mps"
            else:
                state_dict = torch.load(path)
        except Exception as e:
            raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_s{self.d_sae}"
        return sae_name


    @torch.no_grad()
    def resample_neurons_anthropic(self, dead_neuron_indices, model, optimizer, activation_store):
        """
        Arthur's version of Anthropic's feature resampling procedure.
        """
        # collect global loss increases, and input activations
        global_loss_increases, global_input_activations = self.collect_anthropic_resampling_losses(
            model, activation_store
        )

        # sample according to losses
        probs = global_loss_increases / global_loss_increases.sum()
        sample_indices = torch.multinomial(
            probs,
            min(len(dead_neuron_indices), probs.shape[0]),
            replacement=False,
        )
        # if we don't have enough samples for for all the dead neurons, take the first n
        if sample_indices.shape[0] < len(dead_neuron_indices):
            dead_neuron_indices = dead_neuron_indices[:sample_indices.shape[0]]

        # Replace W_dec with normalized differences in activations
        self.W_dec.data[dead_neuron_indices, :] = (
            (
                global_input_activations[sample_indices]
                / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
            )
            .to(self.dtype)
            .to(self.device)
        )
        
        # Lastly, set the new weights & biases
        self.W_enc.data[:, dead_neuron_indices] = self.W_dec.data[dead_neuron_indices, :].T
        self.b_enc.data[dead_neuron_indices] = 0.0
        
        # Reset the Encoder Weights
        if dead_neuron_indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(dead_neuron_indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(dead_neuron_indices))
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale * average_norm

            # Set biases to resampled value
            relevant_biases = self.b_enc.data[dead_neuron_indices].mean()
            self.b_enc.data[dead_neuron_indices] = relevant_biases * 0 # bias resample factor (put in config?)

        else:
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale
            self.b_enc.data[dead_neuron_indices] = -5.0
        
        # TODO: Refactor this resetting to be outside of resampling.
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters

        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, dead_neuron_indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][dead_neuron_indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_out)
                    v[v_key][dead_neuron_indices, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_out,)
                else:
                    if not self.cfg.different_output and not self.cfg.use_gated_sparse_autoencoder:
                        raise ValueError(f"Unexpected dict_idx {dict_idx}")
                        # if we're a transcoder, then this is fine, because we also have b_dec_out
                
        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, dead_neuron_indices].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )
        
        return 

    @torch.no_grad()
    def collect_anthropic_resampling_losses(self, model, activation_store):
        """
        Collects the losses for resampling neurons (anthropic)
        """
        
        batch_size = self.cfg.store_batch_size
        
        # we're going to collect this many forward passes
        number_final_activations = self.cfg.resample_batches * batch_size
        # but have seq len number of tokens in each
        number_activations_total = number_final_activations * self.cfg.context_size
        anthropic_iterator = range(0, number_final_activations, batch_size)
        anthropic_iterator = tqdm(anthropic_iterator, desc="Collecting losses for resampling...")
        
        global_loss_increases = torch.zeros((number_final_activations,), dtype=self.dtype, device=self.device)
        global_input_activations = torch.zeros((number_final_activations, self.d_in), dtype=self.dtype, device=self.device)

        for refill_idx in anthropic_iterator:
            
            # get a batch, calculate loss with/without using SAE reconstruction.
            batch_tokens = activation_store.get_batch_tokenized_data()
            ce_loss_with_recons = self.get_test_loss(batch_tokens, model)
            ce_loss_without_recons, normal_activations_cache = model.run_with_cache(
                batch_tokens,
                names_filter=self.cfg.hook_point,
                return_type = "loss",
                loss_per_token = True,
            )
            # ce_loss_without_recons = model.loss_fn(normal_logits, batch_tokens, True)
            # del normal_logits
            
            normal_activations = normal_activations_cache[self.cfg.hook_point]
            if self.cfg.hook_point_head_index is not None:
                normal_activations = normal_activations[:,:,self.cfg.hook_point_head_index]

            # calculate the difference in loss
            changes_in_loss = ce_loss_with_recons - ce_loss_without_recons
            changes_in_loss = changes_in_loss.cpu()
            
            # sample from the loss differences
            probs = F.relu(changes_in_loss) / F.relu(changes_in_loss).sum(dim=1, keepdim=True)
            changes_in_loss_dist = Categorical(probs)
            samples = changes_in_loss_dist.sample()
            
            assert samples.shape == (batch_size,), f"{samples.shape=}; {self.cfg.store_batch_size=}"
            
            end_idx = refill_idx + batch_size
            global_loss_increases[refill_idx:end_idx] = changes_in_loss[torch.arange(batch_size), samples]
            global_input_activations[refill_idx:end_idx] = normal_activations[torch.arange(batch_size), samples]
        
        return global_loss_increases, global_input_activations

    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """

        if not self.cfg.different_output:
            head_index = self.cfg.hook_point_head_index
            
            def standard_replacement_hook(activations, hook):
                activations = self.forward(activations)[1].to(activations.dtype)
                return activations
            
            def head_replacement_hook(activations, hook):
                new_actions = self.forward(activations[:,:,head_index])[1].to(activations.dtype)
                activations[:,:,head_index] = new_actions
                return activations
    
            replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
            
            ce_loss_with_recons = model.run_with_hooks(
                batch_tokens,
                return_type="loss",
                fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
            )
        else:
            # TODO: currently, this only works with MLP transcoders
            assert("mlp" in self.cfg.out_hook_point)
            
            old_mlp = model.blocks[self.cfg.hook_point_layer]
            class TranscoderWrapper(torch.nn.Module):
                def __init__(self, transcoder):
                    super().__init__()
                    self.transcoder = transcoder
                def forward(self, x):
                    return self.transcoder(x)[0]
            model.blocks[self.cfg.hook_point_layer].mlp = TranscoderWrapper(self)
            ce_loss_with_recons = model.run_with_hooks(
                batch_tokens,
                return_type="loss"
            )
            model.blocks[self.cfg.hook_point_layer] = old_mlp
        
        return ce_loss_with_recons
        
        
class GatedSparseAutoencoder(SparseAutoencoder):
    """
    Attempted to implement as described in https://arxiv.org/pdf/2404.16014
    """
    def __init__(self, cfg):
        super().__init__(cfg, init_weights=False)
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = int(cfg.d_in * cfg.expansion_factor)
        self.l1_coefficient = cfg.l1_coefficient
        self.dtype = cfg.dtype
        self.device = torch.device(cfg.device)

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
            torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device))
        )
        
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
            torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device))
        )
        with torch.no_grad():
            self.set_decoder_norm_to_unit_norm() # Normalize columns of W_dec

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.b_mag = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.r_mag = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )
        
        self.scaling_factor = nn.Parameter(
            torch.ones(self.d_sae, dtype=self.dtype, device=self.device)
        )
        
    def forward(self, x):
        # Compute the forward pass
        
        x = x.to(self.dtype)

        # Remove decoder bias
        input_activations = self.hook_sae_in(x - self.b_dec)

        W_gate_input = einops.einsum(input_activations, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

        self.hidden_pre_mag = W_gate_input * torch.exp(self.r_mag) + self.b_mag
        self.hidden_post_mag = torch.nn.functional.relu(self.hidden_pre_mag)

        self.hidden_pre_gate = W_gate_input + self.b_enc
        self.hidden_post_gate = (torch.sign(self.hidden_pre_gate) + 1) / 2

        self.feature_activations = self.hidden_post_mag * self.hidden_post_gate

        self.output_activations = einops.einsum(self.feature_activations, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec

        return self.feature_activations, self.output_activations


def load_saved_sae(file_path):
    if torch.backends.mps.is_available():
        cfg = torch.load(file_path, map_location="mps")["cfg"]
        cfg.device = "mps"
    elif torch.cuda.is_available():
        cfg = torch.load(file_path, map_location="cuda")["cfg"]
    else:
        cfg = torch.load(file_path, map_location="cpu")["cfg"]

    sparse_autoencoder = SparseAutoencoder.load_from_pretrained(file_path)

    return sparse_autoencoder
