import os
import pandas as pd
import numpy as np
import yaml
import subprocess as sp
import wandb
import math
from functools import partial

import torch
import torch.optim.lr_scheduler as lr_scheduler



def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



@torch.no_grad()
def compute_metrics(input_activations, feature_activations, output_activations):
    """Computes metrics"""

    metrics = {}
    
    # L0
    l0 = (feature_activations > 0).float().sum(-1).mean()
    metrics['l0'] = l0

    # Variance explained
    per_token_l2_loss = (output_activations - input_activations).pow(2).sum(dim=-1).squeeze()
    total_variance = (input_activations - input_activations.mean(0)).pow(2).sum(-1)
    explained_variance = 1 - per_token_l2_loss / total_variance
    unexplained_variance = 1 - explained_variance
    metrics['explained_variance'] = explained_variance.mean()
    metrics['unexplained_variance'] = unexplained_variance.mean()
    
    # Old calculation method to compare with Joseph Bloom's models
    per_token_l2_loss = (output_activations - input_activations).pow(2).sum(dim=-1).squeeze()
    total_variance = input_activations.pow(2).sum(-1)
    metrics['old_explained_variance'] = (1 - per_token_l2_loss/total_variance).mean()

    return metrics



def create_config_inputs(inputs):

    # Dataset
    if 'dataset' in inputs:
        if inputs['dataset'] == "c4-tokenized-2b":
            inputs['dataset_path'] = "NeelNanda/c4-tokenized-2b"
            inputs['is_dataset_tokenized'] = True
        elif inputs['dataset'] == "openwebtext":
            inputs['dataset_path'] = "Skylion007/openwebtext"
            inputs['is_dataset_tokenized'] = False

    # Rename cached_activations_path if not defined
    if 'cached_activations_path' not in inputs or inputs['cached_activations_path'] is None:
        # folder1 = inputs['dataset_path'].replace('/', '_')
        # folder2 = inputs['model_name'].replace('/', '_')
        # inputs['cached_activations_path'] = f"activations/{folder1}/{folder2}/{inputs['hook_point']}"

        # if inputs['hook_point_head_index'] is not None:
        #     inputs['cached_activations_path'] += f"_{inputs['hook_point_head_index']}"
        pass
        
    elif 'cached_activations_path' in inputs:
        inputs['use_cached_activations'] = True

    if 'total_wandb_updates' not in inputs:
        inputs['total_wandb_updates'] = 50
    
    # Auto set d_in if not flattened
    if 'd_in' not in inputs:
        if 'flatten_activations_over_layer' not in inputs or not inputs['flatten_activations_over_layer']:
            attn_hooks = ('hook_q', 'hook_k', 'hook_z', 'hook_v')
            if inputs['hook_point'].endswith(attn_hooks):
                inputs['d_in'] = 64
            elif inputs['hook_point'].endswith('mlp_out'):
                inputs['d_in'] = 512
            

    # Setting total training steps (or total training tokens)
    if 'use_cached_activations' in inputs and inputs['use_cached_activations']:

        # Check cached parameters are the same as inputs
        checks = ['model_name', 'hook_point', 'hook_point_layer', 'hook_point_head_index']
        try:
            with open(inputs['cached_activations_path'] + '/summary.yaml', "r") as yaml_file:
                summary = yaml.safe_load(yaml_file)
                n_activations_on_disk = summary['total_tokens']
        except OSError:
            raise OSError("Couldn't open cached activations")
            
        for x in checks:
            assert summary[x] == inputs[x], ("Cached " + str(x) + "(" + str(summary[x]) + ") does not match input (" + str(inputs[x]) + ")")


        max_allowed_steps = (n_activations_on_disk - ((inputs['n_batches_in_store_buffer'] // 2) * inputs['store_batch_size'] * inputs['context_size'])) // inputs['train_batch_size']


        # Train on all cached activations by default
        if 'train_on_all_cached_activations' not in inputs or inputs['train_on_all_cached_activations']:
            inputs['total_training_steps'] = max_allowed_steps
            inputs['total_training_tokens'] = inputs['total_training_steps'] * inputs['train_batch_size']
            # inputs['wandb_log_frequency'] = inputs['total_training_steps'] // inputs['total_wandb_updates']
            # log("Training on all cached activations")
            # log("Total training steps:", max_allowed_steps)

        # Otherwise raise warning if there are not enough activations
        else:
            # Check that we have enough data on disk
            extra_activations_required = ((inputs['n_batches_in_store_buffer'] // 2) * inputs['store_batch_size'] * inputs['context_size'])
            total_activations_required = inputs['total_training_tokens'] + extra_activations_required
                        
            if n_activations_on_disk < total_activations_required:

                print(f"Only {n_activations_on_disk} activations on disk, but total training tokens required is {total_activations_required}.")
    
                log("Max allowed training steps with cached activations", max_allowed_steps)
                log()            
                
    elif 'total_training_tokens' in inputs:
        inputs['total_training_steps'] = math.ceil(inputs['total_training_tokens'] / inputs['train_batch_size'])
        log("Total training steps:", inputs['total_training_steps'])
        inputs['wandb_log_frequency'] = inputs['total_training_steps'] // inputs['total_wandb_updates']

    # ---------- END SETTING TOTAL TRAINING STEPS ---------- #


    # Make sure the expansion factor is explicit
    if 'expansion_factor' not in inputs:
        log("Warning: Expansion factor not in inputs")
    
    # Learning rate warm up steps - set to 10% of total
    if 'lr_warm_up_steps' not in inputs:
        inputs['lr_warm_up_steps'] = inputs['total_training_steps'] // 10

    if 'train_batch_size' not in inputs:
        log("Warning: train_batch_size not in inputs")

    if 'feature_sampling_window' not in inputs:
        inputs['feature_sampling_window'] = inputs['total_training_steps'] // 20
        inputs['feature_sampling_window'] = inputs['total_training_steps'] // 40
    
    if 'wandb_log_frequency' not in inputs:   
        inputs['wandb_log_frequency'] =  inputs['total_training_steps'] // 50

    if 'eval_frequency' not in inputs:
        inputs['eval_frequency'] =  10 * (inputs['total_training_steps'] // 50)

    
    remove_vals = ['dataset', 'total_wandb_updates', 'total_training_tokens']

    inputs = {k: v for k, v in inputs.items() if k not in remove_vals}    

    return inputs
                   

def log(*message):
    print(*message)


def round_1000(x):
    if x < 1e3:
        return str(x) + ''
    elif x < 1e6:
        return "{:.1f}".format(x / 1e3) + 'K'
    elif x < 1e9:
        return "{:.1f}".format(x / 1e6) + 'M'
    elif x < 1e12:
        return "{:.1f}".format(x / 1e9) + 'G'
    elif x < 1e15:
        return "{:.1f}".format(x / 1e12) + 'T'
    else:
        return "{:.1f}".format(x / 1e15) + 'P'


def log_config(cfg):
    disk_size_per_activation = 4 # Bytes per float32
    

    log("Language Model:")
    # log("Model:", cfg.model_name)
    hook_point = ".".join(cfg.hook_point.split(".")[2:])
    if cfg.hook_point_head_index is None:
        lxhx = "L" + str(cfg.hook_point_layer)
    elif cfg.hook_point_head_index is not None:
        lxhx = "L" + str(cfg.hook_point_layer) + "H" + str(cfg.hook_point_head_index)
    if cfg.flatten_activations_over_layer:
        lxhx = lxhx + ' (flattened)'
    log(cfg.model_name, "|", hook_point, "|", lxhx)
    log()

    log("Dataset:")
    if not cfg.use_cached_activations:
        log("Data", cfg.dataset_path)
    else:
        try:
            with open(cfg.cached_activations_path + '/summary.yaml', "r") as yaml_file:
                summary = yaml.safe_load(yaml_file)
                hook_point = ".".join(summary['hook_point'].split(".")[2:])
                if cfg.hook_point_head_index is None:
                    lxhx = "L" + str(summary['hook_point_layer'])
                elif cfg.hook_point_head_index is not None:
                    lxhx = "L" + str(summary['hook_point_layer']) + "H" + str(summary['hook_point_head_index'])
                if cfg.flatten_activations_over_layer:
                    lxhx = lxhx + ' (flattened)'
            log("Cache - " + cfg.cached_activations_path)
            log(summary['model_name'], "|", hook_point, "|", lxhx)

        except OSError:
            raise OSError("Couldn't open cached activations")    
    log()
    
    log("Sparse Autoencoder:")
    training = str(cfg.train_batch_size) + " batch | " +  round_1000(cfg.total_training_steps) + " steps | " + str(round_1000(cfg.total_training_steps * cfg.train_batch_size)) + " total"
    size = str(cfg.d_in) + " in | " + str(cfg.expansion_factor) + "x | " + str(cfg.d_in * cfg.expansion_factor) + " features"
    log(size)
    log(training)
    log("Learning Rate: " + str(cfg.lr), "|", cfg.lr_scheduler_name, "|", str(cfg.lr_warm_up_steps) + " warm up steps")
    log("L1: " + str(cfg.l1_coefficient))
    log()

    
    log("Activation Store:")
    cfg.tokens_per_buffer = cfg.context_size * cfg.store_batch_size * cfg.n_batches_in_store_buffer
    cfg.tokens_per_store_batch = cfg.context_size * cfg.store_batch_size
    n_buffers = math.ceil((cfg.total_training_steps * cfg.train_batch_size) * 2 / (cfg.context_size * cfg.store_batch_size * cfg.n_batches_in_store_buffer))
    disk_size_per_buffer = round_1000(cfg.tokens_per_buffer * cfg.d_in * disk_size_per_activation) + "b"
    disk_size_per_store_batch = round_1000(cfg.tokens_per_store_batch * cfg.d_in * disk_size_per_activation) + "b"
    log(round_1000(cfg.tokens_per_buffer) + " tokens | " + disk_size_per_buffer, "per buffer", "|", disk_size_per_store_batch, "per batch")
    log()


    

    cfg.total_wandb_updates = cfg.total_training_steps // cfg.wandb_log_frequency

    log("Wandb updates:",  str(cfg.wandb_log_frequency), "steps |", cfg.total_wandb_updates, "windows")

    log(str(cfg.n_checkpoints), "total checkpoints")



@torch.no_grad()
def run_evals(sparse_autoencoder, activation_store, model, n_training_steps):
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

    ### Evals
    eval_tokens = activation_store.get_batch_tokenized_data()

    max_batch_size = 64
    eval_tokens = eval_tokens[:max_batch_size]

    # Get Reconstruction Score
    losses_df = recons_loss_batched(
        sparse_autoencoder, model, activation_store, n_batches = 40,
    )

    recons_score = losses_df["score"].mean()
    ntp_loss = losses_df["loss"].mean()
    recons_loss = losses_df["recons_loss"].mean()
    zero_abl_loss = losses_df["zero_abl_loss"].mean()

    # get cache
    # _, cache = model.run_with_cache(
    #     eval_tokens,
    #     prepend_bos=False,
    #     names_filter=[get_act_name("pattern", hook_point_layer), hook_point],
    # )

    # get act
    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        original_act = cache[sparse_autoencoder.cfg.hook_point][
            :, :, sparse_autoencoder.cfg.hook_point_head_index
        ]
    else:
        original_act = cache[sparse_autoencoder.cfg.hook_point]

    feature_acts, sae_out = sparse_autoencoder(original_act)
    # patterns_original = (
    #     cache[get_act_name("pattern", hook_point_layer)][:, hook_point_head_index]
    #     .detach()
    #     .cpu()
    # )
    del cache

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in

    wandb.log(
        {
            # l2 norms
            "metrics/l2_norm": l2_norm_out.mean().item(),
            "metrics/l2_ratio": l2_norm_ratio.mean().item(),
            # CE Loss
            "metrics/cross_entropy_loss_score": recons_score,
            "metrics/cross_entropy_loss_without_sae": ntp_loss,
            "metrics/cross_entropy_loss_with_sae": recons_loss,
            "metrics/cross_entropy_loss_with_ablation": zero_abl_loss,
        },
        step=n_training_steps,
    )

    head_index = sparse_autoencoder.cfg.hook_point_head_index

    # def standard_replacement_hook(activations, hook):
    #     activations = sparse_autoencoder.forward(activations)[1].to(activations.dtype)
    #     return activations

    # def head_replacement_hook(activations, hook):
    #     new_actions = sparse_autoencoder.forward(activations[:, :, head_index])[1].to(
    #         activations.dtype
    #     )
    #     activations[:, :, head_index] = new_actions
    #     return activations

    # head_index = sparse_autoencoder.cfg.hook_point_head_index
    # replacement_hook = (
    #     standard_replacement_hook if head_index is None else head_replacement_hook
    # )

    # def replacement_hook(activations, hook):
    #     log(activations.shape, "act shape")
    #     activations = sparse_autoencoder.forward(activations)[1].to(activations.dtype)
    #     log(activations.shape, "act shape")
    #     return activations
        
    # get attn when using reconstructed activations
    # with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
    #     _, new_cache = model.run_with_cache(
    #         eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
    #     )
    #     patterns_reconstructed = (
    #         new_cache[get_act_name("pattern", hook_point_layer)][
    #             :, hook_point_head_index
    #         ]
    #         .detach()
    #         .cpu()
    #     )
    #     del new_cache

    # # get attn when using reconstructed activations
    # with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
    #     _, zero_ablation_cache = model.run_with_cache(
    #         eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
    #     )
    #     patterns_ablation = (
    #         zero_ablation_cache[get_act_name("pattern", hook_point_layer)][
    #             :, hook_point_head_index
    #         ]
    #         .detach()
    #         .cpu()
    #     )
    #     del zero_ablation_cache

    # if sparse_autoencoder.cfg.hook_point_head_index:
    #     kl_result_reconstructed = kl_divergence_attention(
    #         patterns_original, patterns_reconstructed
    #     )
    #     kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()

    #     kl_result_ablation = kl_divergence_attention(
    #         patterns_original, patterns_ablation
    #     )
    #     kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()

    #     if wandb.run is not None:
    #         wandb.log(
    #             {
    #                 "metrics/kldiv_reconstructed": kl_result_reconstructed.mean().item(),
    #                 "metrics/kldiv_ablation": kl_result_ablation.mean().item(),
    #             },
    #             step=n_training_steps,
    #         )


def recons_loss_batched(sparse_autoencoder, model, activation_store, n_batches=10, max_batch_size=256):
    losses = []
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokenized_data()
        batch_tokens = batch_tokens[:max_batch_size]
        score, loss, recons_loss, zero_abl_loss = get_recons_loss(
            sparse_autoencoder, model, batch_tokens
        )
        losses.append(
            (
                score.mean().item(),
                loss.mean().item(),
                recons_loss.mean().item(),
                zero_abl_loss.mean().item(),
            )
        )

    losses = pd.DataFrame(
        losses, columns=["score", "loss", "recons_loss", "zero_abl_loss"]
    )

    return losses



# @torch.no_grad()
def get_recons_loss(sparse_autoencoder, model, batch_tokens):
    hook_point = sparse_autoencoder.cfg.hook_point
    loss = model(batch_tokens, return_type="loss")
    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[1].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[1].to(activations.dtype)
        activations[:, :, head_index] = new_activations
        return activations

    replacement_hook = (
        standard_replacement_hook if head_index is None else head_replacement_hook
    )
    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
    )

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def zero_ablate_hook(activations, hook):
    activations = torch.zeros_like(activations)
    return activations


def lr_constant(steps):
    return 1

def lr_constant_with_warmup(steps, warm_up_steps):
    return min(1.0, (steps + 1) / warm_up_steps)

def lr_linear_warmup_decay(steps, warm_up_steps, total_steps):
    if steps < warm_up_steps:
        return (steps + 1) / warm_up_steps
    else:
        return (total_steps - steps) / (total_steps - warm_up_steps)

def lr_linear_warmup_cosine_decay(steps, warm_up_steps, total_steps, lr_end):
    if steps < warm_up_steps:
        return (steps + 1) / warm_up_steps
    else:
        progress = (steps - warm_up_steps) / (total_steps - warm_up_steps)
        return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))


def get_scheduler(cfg, optimizer):
    """
    Args:
        scheduler_name (Optional[str]): Name of the scheduler to use. If None, returns a constant scheduler
        optimizer (optim.Optimizer): Optimizer to use
    """
    scheduler_name = cfg.lr_scheduler_name
    warm_up_steps = cfg.lr_warm_up_steps
    total_steps = cfg.total_training_steps
    lr_end = 0

    schedulers = {}
    
    schedulers['constant'] = lr_constant
    schedulers['constant_with_warmup'] = partial(lr_constant_with_warmup, warm_up_steps=warm_up_steps)
    schedulers['linear_warmup_decay'] = partial(lr_linear_warmup_decay, warm_up_steps=warm_up_steps, total_steps=total_steps)
    schedulers['linear_warmup_cosine_decay'] = partial(lr_linear_warmup_cosine_decay, warm_up_steps=warm_up_steps, total_steps=total_steps, lr_end=lr_end)
    schedulers['cosine_annealing'] = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_end)

    lr_function = schedulers[scheduler_name]

    return lr_scheduler.LambdaLR(optimizer, lr_function)


def get_blog_checkpoint(wandb_run_name):
    checkpoint_dir = "../outputs/blog_1_checkpoints"
    checkpoint_path = checkpoint_dir + "/" + wandb_run_name + "_final_checkpoint_new.pt"
    return checkpoint_path

def get_blog_sparsity(wandb_run_name):
    checkpoint_dir = "../outputs/blog_1_checkpoints"
    checkpoint_path = checkpoint_dir + "/" + wandb_run_name + "_final_sparsity.pt"
    return checkpoint_path





def latest_feature_sparsity_name(wandb_name, checkpoint_dir="checkpoints", i_checkpoint=-1):
    def numeric_part(filename):
        return int(filename.split('_')[-4])
        
    if type(wandb_name) == int:
        wandb_runs = os.listdir(checkpoint_dir)
        filename = next((modname for modname in wandb_runs if modname.endswith('-' + str(wandb_name))), None)
        wandb_name = filename
    
    
    checkpoint_path = checkpoint_dir + '/' + wandb_name
    file_extension = 'log_feature_sparsity.pt'
    files = [filename for filename in os.listdir(checkpoint_path) if filename.endswith(file_extension) and not filename.endswith('final_log_feature_sparsity.pt')]
    
    sorted_files = sorted(files, key=numeric_part)
    
    checkpoint_name_dir = sorted_files[i_checkpoint]
    
    checkpoint_name = checkpoint_path + '/' + checkpoint_name_dir

    return checkpoint_name

def latest_checkpoint_name(wandb_name, checkpoint_dir="checkpoints", i_checkpoint=-1):

    def numeric_part(filename):
        return int(filename.split('_')[-1][:-3])
        
    if type(wandb_name) == int:
        wandb_runs = os.listdir(checkpoint_dir)
        filename = next((modname for modname in wandb_runs if modname.endswith('-' + str(wandb_name))), None)
        wandb_name = filename
    
    
    checkpoint_path = checkpoint_dir + '/' + wandb_name
    file_extension = '.pt'
    files = [filename for filename in os.listdir(checkpoint_path) if filename.endswith(file_extension) and not filename.endswith('sparsity.pt')]
    sorted_files = sorted(files, key=numeric_part)
    
    checkpoint_name_dir = sorted_files[i_checkpoint]
    
    # checkpoint_name = checkpoint_path + '/' + checkpoint_name_dir

    return checkpoint_path, checkpoint_name_dir

def create_lineplot_histogram(distribution, bins=20):
    vals, bin_edges = np.histogram(distribution, bins=bins)

    xvals = np.repeat(bin_edges, 2)
    yvals = np.repeat(vals, 2)
    yvals = np.concatenate(([0], yvals, [0]))

    return xvals, yvals
    
















