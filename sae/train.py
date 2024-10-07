import torch
from torch.optim import Adam
import torch.nn.functional as F

import gc
import os
import numpy as np
import yaml
import einops
from tqdm.auto import tqdm
from dataclasses import asdict
import wandb
from transformer_lens import HookedTransformer

from sae.utils import get_scheduler, log
from sae.metrics import compute_simple_metrics, compute_recovered_loss, compute_recovered_loss_transcoder
from sae.sparse_autoencoder import SparseAutoencoder, GatedSparseAutoencoder, load_saved_sae
from sae.activation_store import ActivationsStore, DoubleActivationStore


class ModelTrainer():
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
    
    def setup(self, activation_store=None, model=None):
        """
        Loads model, Activation Store, Sparse Autoencoder
        """
    
        # Load model
        if model is None:
            self.model = HookedTransformer.from_pretrained(self.cfg.model_name)
            self.model.to(self.device)
        else:
            self.model = model

        # Create activation store
        log("creating activation store")
        if self.cfg.different_output:
            self.activation_store = DoubleActivationStore(self.cfg, self.model)
        elif activation_store is None:
            self.activation_store = ActivationsStore(self.cfg, self.model)
        else:
            self.activation_store = activation_store

        # Create Sparse Autoencoder
        log("creating sae")
        if self.cfg.from_pretrained_path is not None:
            self.sparse_autoencoder = load_saved_sae(self.cfg.from_pretrained_path)
            loaded_cfg = self.sparse_autoencoder.cfg
            if self.cfg.n_starting_steps is None:
                self.n_starting_steps = int(self.cfg.from_pretrained_path.split('_')[-1][:-3]) // self.cfg.train_batch_size
            else:
                self.n_starting_steps = self.cfg.n_starting_steps
            self.true_feature_sparsity_window = torch.zeros(loaded_cfg.n_running_sparsity, loaded_cfg.d_in * loaded_cfg.expansion_factor).to(loaded_cfg.device)
            
            if self.cfg.fine_tune_dataset:
                loaded_cfg.dataset = self.cfg.dataset
                loaded_cfg.is_dataset_tokenized = self.cfg.is_dataset_tokenized
                loaded_cfg.use_cached_activations = self.cfg.use_cached_activations
                loaded_cfg.cached_activations_path = self.cfg.cached_activations_path
                loaded_cfg.loop_dataset = self.cfg.loop_dataset
                loaded_cfg.n_batches_in_store_buffer = self.cfg.n_batches_in_store_buffer
                loaded_cfg.store_batch_size = self.cfg.store_batch_size
                loaded_cfg.total_training_steps = self.n_starting_steps + self.cfg.finetuning_steps
            self.cfg = loaded_cfg

        elif self.cfg.use_gated_sparse_autoencoder:
            self.sparse_autoencoder = GatedSparseAutoencoder(self.cfg)
            self.n_starting_steps = 0
            
        else:
            self.sparse_autoencoder = SparseAutoencoder(self.cfg)
            self.n_starting_steps = 0
            
        # Set decoder to mean activation
        if self.cfg.b_dec_init_method == "mean":
            self.sparse_autoencoder.initialize_b_dec_with_mean(self.activation_store)
   
        if self.cfg.scale_input_norm:
            scaling_factor = calculate_constant_input_scaling_factor(self.activation_store)
            self.sparse_autoencoder.input_scaling_factor = scaling_factor


        # Creating wandb
        log("creating wanbd")
        if self.cfg.log_to_wandb and not self.cfg.multiple_runs:
            wandb.init(project=self.cfg.wandb_project, config=self.cfg)
            self.cfg.full_checkpoint_path = self.cfg.checkpoint_path + "/" + wandb.run.name
        else:
            unique_id = wandb.util.generate_id()
            self.cfg.full_checkpoint_path = self.cfg.checkpoint_path + "/" + unique_id
        os.makedirs(self.cfg.full_checkpoint_path)
        
        # Save hyperparameters
        self.save_hyperparameters() 

        # Setup optimizer
        self.optimizer = Adam(self.sparse_autoencoder.parameters(),
                              lr=self.cfg.lr,
                              betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))

        # Create learning rate scheduler
        self.scheduler = get_scheduler(self.cfg, self.optimizer)

        if self.cfg.l1_warmup:
            self.final_l1_coefficient = self.cfg.l1_coefficient

        if self.cfg.l0_warmup:
            self.final_l0_loss_coefficient = self.cfg.l0_coefficient

        self.n_training_steps = 0 + self.n_starting_steps
        self.n_training_tokens = 0
        self.total_training_steps = self.cfg.total_training_steps
        
    def train(self):
        """
        Main training loop
        """
        for n_training_steps in tqdm(range(self.n_training_steps, self.total_training_steps)):
            
            gc.collect()
            torch.cuda.empty_cache()
            
            self.input_activations = self.get_input_activations()
            self.do_training_step()

        
        self.save_checkpoint(self.cfg)
        self.save_sparsity_distribution(self.cfg)
    
    def get_input_activations(self):
        """
        Gets input activations within a given training step
        """
        if not self.cfg.different_output:
            self.input_activations = self.activation_store.next_batch()
            self.target_activations = self.input_activations
        else:
            self.input_activations, self.target_activations = self.activation_store.next_batch()

        return self.input_activations
    
    def do_training_step(self):
        """
        Does training step given input_activations
        """
        self.run_before_step() # Warms up coefficients & resampling
        
        self.optimizer.zero_grad()

        self.feature_activations, self.output_activations = self.sparse_autoencoder(self.input_activations)
        
        self.compute_loss()
        self.loss.backward()

        self.run_after_loss() # clip grad norm and normalise decoder

        self.optimizer.step()
        self.scheduler.step()
        
        self.run_after_step() # Logs outputs to wandb & saves checkpoints


    def run_before_step(self):
        """
        Warm up values of loss coefficients
        """
        if self.cfg.l1_warmup:
            if self.n_training_steps < self.cfg.l1_warmup_steps:
                self.cfg.l1_coefficient = self.final_l1_coefficient * (self.n_training_steps)/(self.cfg.l1_warmup_steps)
            else:
                self.cfg.l1_coefficient = self.final_l1_coefficient

        if self.cfg.l0_warmup:
            if self.n_training_steps < self.cfg.l0_warmup_steps:
                self.cfg.l0_loss_coefficient = self.final_l0_loss_coefficient * (self.n_training_steps)/(self.cfg.l0_warmup_steps)
            else:
                self.cfg.l0_loss_coefficient = self.final_l0_loss_coefficient

        # Do resampling if needed
        if self.cfg.feature_resampling_method == 'anthropic':
            if ((self.n_training_steps + 1) % self.cfg.resample_frequency == 0) and self.n_training_steps < self.cfg.max_resample_step:
                dead_neuron_indices = (self.loss_sparsity <= self.cfg.min_sparsity_for_resample).nonzero(as_tuple=False)[:, 0]
                self.num_neurons_resampled = len(dead_neuron_indices)
                if self.num_neurons_resampled > 0:
                    print("Number of neurons resampled:", self.num_neurons_resampled)
                    self.sparse_autoencoder.resample_neurons_anthropic(dead_neuron_indices, self.model, self.optimizer, self.activation_store)
                else:
                    print("No neurons resampled")

    
    def run_after_loss(self):
        if self.cfg.clip_grad_norm:
            clip_grad_norm_(self.sparse_autoencoder.parameters(), max_norm=1.0)

        if self.cfg.normalise_w_dec:
            self.sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()

    
    def run_after_step(self):
        # Normalize W_dec if needed
        if self.cfg.normalise_w_dec:
            self.sparse_autoencoder.set_decoder_norm_to_unit_norm()
            
        # Save if at checkpoint frequency
        if self.cfg.n_checkpoints > 0 and ((self.n_training_steps+1) % (self.total_training_steps // self.cfg.n_checkpoints) == 0):
            self.save_checkpoint(self.cfg)
        
        # Save if at sparsity frequency
        if (self.n_training_steps + 1) % self.cfg.sparsity_log_frequency == 0:
            self.save_sparsity_distribution(self.cfg)
        
        if self.cfg.log_to_wandb or self.cfg.multiple_runs:
            if ((self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0) or self.n_training_steps < 10:
                self.log_during_step()

        self.n_training_tokens += self.cfg.train_batch_size
        self.n_training_steps += 1

    
    @torch.no_grad()
    def log_during_step(self):
        self.log_vals = {}

        # Logging main metrics
        metrics = compute_simple_metrics(self.target_activations, self.feature_activations, self.output_activations)        
        metrics['learning_rate'] = self.optimizer.param_groups[0]["lr"]

        # Logging sparsity
        metrics['true_feature_sparsity_hist'] = wandb.Histogram(torch.log10(self.true_feature_sparsity + 1e-10).detach().cpu().numpy())

        sparsity_thresholds = [0.3, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 5e-6, 1e-6, 1e-8]
        sparsity_strings = ['3e-1', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6', '1e-8']
        
        for threshold, name in zip(sparsity_thresholds, sparsity_strings):
            metrics['true_sparsity_below_' + name] = (self.true_feature_sparsity < threshold).sum().item()
            metrics['true_sparsity_above_' + name] = (self.true_feature_sparsity > threshold).sum().item()

        # Convert for wandb format
        metrics  = {"metrics/" + key: value for key, value in metrics.items()}
        losses  = {"loss/" + key: value for key, value in self.loss_values.items()}

        self.log_vals.update(losses)
        self.log_vals.update(metrics)

        if ((self.n_training_steps + 1) % self.cfg.eval_frequency) == 0:

            if self.cfg.different_output:
                reconstruction_metrics = compute_recovered_loss_transcoder(self.sparse_autoencoder,
                                                                           self.activation_store,
                                                                           self.model,
                                                                           self.cfg.hook_point,
                                                                           self.cfg.hook_point_output,
                                                                           self.cfg.hook_point_head_index)
            else:           
                reconstruction_metrics = compute_recovered_loss(self.sparse_autoencoder,
                                                                self.activation_store,
                                                                self.model,
                                                                self.cfg.hook_point,
                                                                self.cfg.hook_point_head_index)

            reconstruction_metrics  = {"metrics/" + key: value for key, value in reconstruction_metrics.items()}
            self.log_vals.update(reconstruction_metrics)
            
        if self.cfg.feature_resampling_method == 'anthropic':
            if ((self.n_training_steps + 1) % self.cfg.resample_frequency == 0) and self.n_training_steps < self.cfg.max_resample_step:
                self.log_vals.update({"metrics/num_neurons_resampled": self.num_neurons_resampled})

        # Log metrics
        if self.cfg.log_to_wandb:
            wandb.log(self.log_vals, step=self.n_training_steps)
            
    
    def save_hyperparameters(self):
        with open(self.cfg.full_checkpoint_path + '/hyperparameters.yaml', 'w') as yaml_file:
            dict_vals = asdict(self.cfg)
            dict_vals['device'] = 'cuda'
            dict_vals['dtype'] = 'float32'
            yaml.dump(dict_vals, yaml_file)
            
    def save_sparsity_distribution(self, cfg):
        main_path = f"{cfg.full_checkpoint_path}/{self.sparse_autoencoder.get_name()}_{self.n_training_tokens}"
        torch.save(torch.log10(self.true_feature_sparsity + 1e-10), main_path + "_log_feature_sparsity.pt")
    
    def save_checkpoint(self, cfg):
        main_path = f"{cfg.full_checkpoint_path}/{self.sparse_autoencoder.get_name()}_{self.n_training_tokens}"
        self.sparse_autoencoder.save_model(main_path + ".pt")


    def compute_running_sparsity(self):
        """
        Computes the running sparsity over the last cfg.n_running_sparsity training steps
        by storing and updating self.true_feature_sparsity_window each step
        """
        self.true_batch_feature_sparsity = (self.feature_activations > 0).float().mean(dim=0)
        n_average = self.cfg.n_running_sparsity

        if self.n_training_steps == 0:
            self.true_feature_sparsity_window = torch.zeros(n_average, self.cfg.d_in * self.cfg.expansion_factor).to(self.cfg.device)

        self.true_feature_sparsity_window[self.n_training_steps % n_average] = self.true_batch_feature_sparsity.detach()

        n_end = min(n_average, self.n_training_steps + 1)
        existing_means = self.true_feature_sparsity_window[:n_end].mean(dim=1) # Check for outliers
        close_means = (existing_means.log10() - self.true_batch_feature_sparsity.mean().log10()).abs() < 0.5

        self.true_feature_sparsity = self.true_feature_sparsity_window[:n_end][close_means].mean(dim=0)

        # # Approximate feature sparsity calculations 
        self.activations_bool = self.feature_activations / (self.feature_activations + self.cfg.epsilon_l0_approx)
        self.approx_batch_feature_sparsity = self.activations_bool.mean(dim=0) 
        self.loss_sparsity = self.approx_batch_feature_sparsity.clone()

        with torch.no_grad():
            self.loss_sparsity.data = self.true_feature_sparsity.data

    
    def compute_loss(self):
        """
        Computes the loss. self.loss_values is for logging to wandb
        """
        self.compute_running_sparsity()
        self.loss_values = {}
        self.loss = 0

        # Mean squared error
        self.loss += self.compute_mse_loss()

        # L1 penalty
        if self.cfg.l1_coefficient > 0:
            self.loss += self.compute_l1_loss()

        # L0-approx penalty
        if self.cfg.l0_coefficient > 0:
            self.loss += self.compute_l0_approx_loss()

        # Custom method to penalise sparse features
        if self.cfg.sparse_loss_coefficient > 0:
            self.loss += self.compute_min_sparsity_loss()

        # Auxiliary loss for Gated SAE
        if self.cfg.use_gated_sparse_autoencoder:
            self.loss += self.compute_gated_auxiliary_loss()

        return self.loss

    def compute_mse_loss(self):
        """
        Computes mean squared error loss
        """
        # Standard
        if self.cfg.mse_loss_type == "standard":
            mse = torch.nn.functional.mse_loss(self.output_activations, self.target_activations, reduction="none").sum(dim=-1)

        # Centered
        elif self.cfg.mse_loss_type == "centered":
            x_centred = self.target_activations - self.target_activations.mean(dim=0, keepdim=True)
            mse = torch.pow((self.output_activations - self.target_activations), 2) / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
            
        self.unweighted_mse_loss = mse.mean()
        self.mse_loss = self.cfg.mse_loss_coefficient * self.unweighted_mse_loss
               
        self.loss_values['unweighted_mse_loss'] =  self.unweighted_mse_loss.item()
        self.loss_values['mse_loss'] =  self.mse_loss.item()

        return self.mse_loss
        
    def compute_l1_loss(self):
        """
        Computes L1 loss for either:
        1. Gated SAE
        2. Weighted by decoder norm as proposed by Anthropic
        3. Standard L1 norm
        """
        # Gated SAE
        if self.cfg.use_gated_sparse_autoencoder:
            via_gate_feature_magnitudes = F.relu(self.sparse_autoencoder.hidden_pre_gate)        
            self.unweighted_l1_loss = via_gate_feature_magnitudes.norm(p=1, dim=1).mean()

        elif self.cfg.weight_l1_by_decoder_norms and self.cfg.custom_loss == "l0_anthropic":
            w_dec_norms = self.sparse_autoencoder.W_dec.norm(dim=1)
            weighted_feature_acts = self.feature_activations * self.sparse_autoencoder.W_dec.norm(dim=1)
            self.activations_bool = weighted_feature_acts / (weighted_feature_acts + self.cfg.epsilon_l0_approx)
            self.unweighted_l1_loss = self.activations_bool.sum(dim=1).mean()
            
            with torch.no_grad():
                self.unweighted_l1_loss.data = (weighted_feature_acts > 0).float().sum(dim=1).mean()
            
        # Anthropic's version
        elif self.cfg.weight_l1_by_decoder_norms:
            w_dec_norms = self.sparse_autoencoder.W_dec.norm(dim=1)
            weighted_feature_acts = self.feature_activations * self.sparse_autoencoder.W_dec.norm(dim=1)
            self.unweighted_l1_loss = weighted_feature_acts.norm(p=1, dim=-1).mean()
            
        # Standard version
        else:
            self.unweighted_l1_loss = torch.abs(self.feature_activations).sum(dim=1).mean()

        self.l1_loss = self.cfg.l1_coefficient * self.unweighted_l1_loss
        
        self.loss_values['unweighted_l1_loss'] =  self.unweighted_l1_loss.item()
        self.loss_values['l1_loss'] =  self.l1_loss.item() 
        self.loss_values['normal_l1_loss'] = torch.abs(self.feature_activations).sum(dim=1).mean().item()
        self.loss_values['W_dec_l2_norm_mean'] = self.sparse_autoencoder.W_dec.norm(dim=-1).mean().item()
        
        return self.l1_loss

    def compute_l0_approx_loss(self):
        """
        L0-approx penalty loss
        """
        self.activations_bool = self.feature_activations / (self.feature_activations + self.cfg.epsilon_l0_approx)
        self.unweighted_l0_loss = self.activations_bool.sum(dim=1).mean()
        
        with torch.no_grad():
            self.unweighted_l0_loss.data = (self.feature_activations > 0).float().sum(dim=1).mean()                
 
        self.l0_loss = self.cfg.l0_coefficient * self.unweighted_l0_loss

        self.loss_values['unweighted_l0_loss'] =  self.unweighted_l0_loss.item()
        self.loss_values['l0_loss'] =  self.l0_loss.item()

        return self.l0_loss

    def compute_min_sparsity_loss(self):
        """
        Provides ways to limit dead features and discourage dense features
        """
        self.unweighted_sparse_loss = torch.nn.functional.relu(np.log10(self.cfg.min_sparsity_target) -  torch.log10(self.loss_sparsity + 1e-10)).sum()
        self.sparse_loss = self.cfg.sparse_loss_coefficient * self.unweighted_sparse_loss

        self.loss_values['unweighted_sparse_loss'] =  self.unweighted_sparse_loss.item()
        self.loss_values['sparse_loss'] =  self.sparse_loss.item()
        
        self.unweighted_dense_loss = torch.nn.functional.relu(torch.log10(self.loss_sparsity + 1e-10) - np.log10(self.cfg.max_sparsity_target)).sum()
        self.dense_loss = self.cfg.dense_loss_coefficient * self.unweighted_dense_loss

        self.loss_values['unweighted_dense_loss'] =  self.unweighted_dense_loss.item()
        self.loss_values['dense_loss'] =  self.dense_loss.item()

        return self.sparse_loss + self.dense_loss


    def compute_gated_auxiliary_loss(self):
        """
        Auxiliary Loss for the Gated SAEs (https://arxiv.org/pdf/2404.16014)
        """
        via_gate_feature_magnitudes = F.relu(self.sparse_autoencoder.hidden_pre_gate)        

        via_gate_reconstruction = (
            einops.einsum(
                via_gate_feature_magnitudes,
                self.sparse_autoencoder.W_dec.detach(),
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.sparse_autoencoder.b_dec.detach()
        )
        self.aux_loss = torch.nn.functional.mse_loss(via_gate_reconstruction, self.target_activations, reduction="mean")

        self.loss_values['aux_loss'] =  self.aux_loss.item()
        self.loss_values['aux_loss'] =  self.aux_loss.item()

        return self.aux_loss



@torch.no_grad()
def calculate_constant_input_scaling_factor(activation_store, n_batch=5):
    """
    Calculates scaling factor as per Anthropic's April 2024 upate
    """
    l2_norms = []
    
    for i in range(n_batch):
        input_activations = activation_store.next_batch()
        l2_norm = torch.linalg.norm(input_activations, dim=1, ord=2).mean()
        l2_norms.append(l2_norm)

    mean_l2_norm = torch.tensor(l2_norms).mean().item()
    n = np.sqrt(input_activations.shape[-1])
    input_scaling_factor = n/mean_l2_norm
    return input_scaling_factor




class MultipleModelTrainer():
    def __init__(self, cfg_list):
        self.cfg_list = cfg_list
        self.cfg_0 = cfg_list[0]
        self.device = torch.device(self.cfg_0.device)

    def setup(self, activation_store=None):
        self.model = HookedTransformer.from_pretrained(self.cfg_0.model_name)
        self.model.to(self.device)
        
        log("creating activation store")
        if self.cfg_0.different_output:
            self.activation_store = DoubleActivationStore(self.cfg_0, self.model)
        elif activation_store is None:
            self.activation_store = ActivationsStore(self.cfg_0, self.model)
        else:
            self.activation_store = activation_store

        
        self.model_trainer_list = [ModelTrainer(cfg) for cfg in self.cfg_list]

        
        for i, mod in enumerate(self.model_trainer_list):
            mod.setup(activation_store=self.activation_store, model=self.model)
            mod.log_name = "mod" + str(i+1)
        
        log("creating wanbd")
        if self.cfg_0.log_to_wandb:
            wandb.init(project=self.cfg_0.wandb_project, config=self.cfg_0)


    def train(self):
        """
        Main training loop
        """
        self.total_training_steps = self.model_trainer_list[0].total_training_steps
        
        for n_training_steps in tqdm(range(self.total_training_steps)):
            
            self.input_activations = self.get_input_activations()
            
            all_log_vals = {}
            
            for mod in self.model_trainer_list:
                mod.input_activations = self.input_activations
                mod.target_activations = self.target_activations
                mod.do_training_step()
                
                log_vals  = {mod.log_name + "/" + key: value for key, value in mod.log_vals.items()}

                all_log_vals.update(log_vals)

            # Log metrics
            if self.cfg_0.log_to_wandb and ((n_training_steps + 1) % self.cfg_0.wandb_log_frequency == 0):
                wandb.log(all_log_vals, step=n_training_steps)
        
        for mod in self.model_trainer_list:
            mod.save_checkpoint(self.cfg)
            mod.save_sparsity_distribution(self.cfg)

        
    def get_input_activations(self):
        if not self.cfg_0.different_output:
            self.input_activations = self.activation_store.next_batch()
            self.target_activations = self.input_activations
        else:
            self.input_activations, self.target_activations = self.activation_store.next_batch()

        return self.input_activations
        
   











