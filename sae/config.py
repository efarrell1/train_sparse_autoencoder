import math
import torch
import itertools
from dataclasses import dataclass

from sae.utils import round_1000, log, create_config_inputs


@dataclass()
class Config():
    # Model and Hook Point
    model_name: str = "gelu-1l"
    hook_point: str = "blocks.0.hook_mlp_out"
    hook_point_layer: int = 0
    hook_point_head_index: int | None = None
    flatten_activations_over_layer: bool = False
    d_in: int = 2048
    
    # Different output / Transcoder (optional)
    different_output: bool = False
    hook_point_output: int | None = None
    hook_point_layer_output: int | None = None
    hook_point_head_index_output: int | None = None
    flatten_activations_over_layer_output: bool = False
    d_out: int | None = None
    
    # Dataset
    dataset: str | None = None
    is_dataset_tokenized: bool = True
    use_cached_activations: bool = False
    cached_activations_path: str | None = None 
    loop_dataset: bool = False
    fine_tune_dataset: bool = False

    # Activation Store Parameters
    n_batches_in_store_buffer: int = 20
    store_batch_size: int = 32
    train_batch_size: int = 4096
    context_size: int = 128
    remove_bos_tokens: bool = False

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Outputs
    log_to_wandb: bool = True
    wandb_project: str = "test_gpt2_small"
    wandb_log_frequency: int = 10
    eval_frequency: int = 100000
    sparsity_log_frequency: int = 5000
    n_checkpoints: int = 10
    checkpoint_path: str = "../outputs/checkpoints"
    
    # Sparse Autoencoder Parameters
    expansion_factor: int = 1
    b_dec_init_method: str = "zeros"
    subtract_b_dec_from_inputs: bool = False
    from_pretrained_path: str | None = None
    use_gated_sparse_autoencoder: bool = False
    normalise_w_dec: bool = True

    clip_grad_norm: bool = False
    normalise_initial_decoder_weights: bool = False
    initial_decoder_norm: float = 0.1
    initialise_encoder_to_decoder_transpose: bool = False
    scale_input_norm: bool = False

    activation_function: str = "relu"
    topk_amount: int = 10

    # Resampling
    feature_resampling_method: str | None = None # anthropic
    resample_frequency: int = 25000
    max_resample_step: int = 100000
    resample_batches: int = 128
    feature_reinit_scale: float = 0.2
    min_sparsity_for_resample: float = 0

    # General
    seed: int = 42
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    total_training_steps: int = 1000
    n_starting_steps: str | None = None # specify how many steps were taken during pretraining
    finetuning_steps: int = 1000,
    multiple_runs: bool = False

    # Learning rate parameters
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"
    lr_warm_up_steps: int = 500

    # Loss Function
    mse_loss_coefficient: float = 1
    mse_loss_type: str = "standard" # standard or centered
    
    l1_coefficient: float = 0
    weight_l1_by_decoder_norms: bool = False
    
    l0_coefficient: float = 0
    epsilon_l0_approx: float = 0.1

    sparse_loss_coefficient: float = 0
    dense_loss_coefficient: float = 0
    max_sparsity_target: float = 1
    min_sparsity_target: float = 0
    n_running_sparsity: int = 300

    custom_loss: str | None = None
    
    # Warm up loss coefficients
    l1_warmup: bool = False
    l1_warmup_steps: int = 1000
    l0_warmup: bool = False
    l0_warmup_steps: int = 1000



def create_config(inputs):
    inputs = create_config_inputs(inputs)
    return Config(**inputs)

def generate_cfg_list(main_config_inputs, sweep):
    combinations = [dict(zip(sweep.keys(), values)) for values in itertools.product(*sweep.values())]
    
    cfg_list = []
    for combo in combinations:
        specific_inputs = main_config_inputs.copy()
        specific_inputs.update(combo)
        cfg = Config(**specific_inputs)
        cfg.multiple_runs = True
        cfg_list.append(cfg)
    return cfg_list
    
@dataclass
class CachedActivationsConfig():
    # Model
    model_name: str = "gelu-2l"
    hook_point: str = "blocks.0.hook_mlp_out"
    hook_point_layer: int = 0
    hook_point_head_index: int | None = None
    flatten_activations_over_layer: bool = False
    d_in: int = 512
    
    # Dataset
    dataset: str | None = None
    is_dataset_tokenized: bool = True
    context_size: int = 128
    cached_activations_path: str | None = None 
    loop_dataset: bool = False
    loop_cache: bool = False

    # Activation Store Parameters
    n_batches_in_store_buffer: int = 20
    total_training_tokens: int = 1_000_000
    store_batch_size: int = 32
    
    # Cache activations
    shuffle_every_n_buffers: int = 10
    n_shuffles_with_last_section: int = 10
    n_shuffles_in_entire_dir: int = 10
    n_shuffles_final: int = 5
    
    # Misc
    device: str = "cuda"
    seed: int = 42
    dtype: torch.dtype = torch.float32
    use_cached_activations: bool = False

    log_to_wandb: bool = False

    def __post_init__(self, print_config=True):
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            path = f"activations/{self.dataset.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.hook_point}"
            self.cached_activations_path = path
            if self.hook_point_head_index is not None:
                self.cached_activations_path += f"_{self.hook_point_head_index}"


        self.device = torch.device(self.device)

        # self.n_batches_in_store_buffer = math.ceil(self.total_training_tokens / (self.context_size * self.store_batch_size * self.shuffle_every_n_buffers))
        self.tokens_per_buffer = self.context_size * self.store_batch_size * self.n_batches_in_store_buffer

        self.n_contexts_per_buffer = self.store_batch_size * self.n_batches_in_store_buffer
        
        if print_config:
            self.log_config()


    def log_config(self):
        disk_size_per_activation = 4 # Bytes per float32
        
        log("Activation Store")
        n_buffers = math.ceil(self.total_training_tokens / (self.context_size * self.store_batch_size * self.n_batches_in_store_buffer))
        log(f"Total buffers: {n_buffers}")
        log(f"Tokens per batch:", round_1000(self.context_size * self.store_batch_size))    
        log(f"Batches per buffer:", self.n_batches_in_store_buffer)    
        log(f"Tokens per buffer:", round_1000(self.tokens_per_buffer))
        log()
        log(f"Disk size per buffer:", round_1000(self.tokens_per_buffer * self.d_in * disk_size_per_activation) + "b")
        log(f"Total disk size:", round_1000(self.tokens_per_buffer * self.d_in * disk_size_per_activation * n_buffers) + "b")
        log()

