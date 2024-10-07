import torch
import einops
import numpy as np
import pandas as pd
import os
import subprocess as sp

from transformer_lens import utils


def load_sae_from_path(path):
    pass


def make_token_df(model, tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    prefix_list = []
    suffix_list = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            prefix_list.append(prefix)
            suffix_list.append(suffix)
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        prefix=prefix_list,
        suffix=suffix_list,
        batch=batch,
        pos=pos,
        label=label,
    ))

SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def process_tokens_index(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)


def list_flatten(nested_list):
    return [x for y in nested_list for x in y]


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


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
    files = [filename for filename in os.listdir(checkpoint_path) if filename.endswith(file_extension) and not filename.endswith(('_final.pt', 'sparsity.pt'))]
    sorted_files = sorted(files, key=numeric_part)
    
    checkpoint_name_dir = sorted_files[i_checkpoint]
    
    checkpoint_name = checkpoint_path + '/' + checkpoint_name_dir

    return checkpoint_name


def replacement_hook(acts, hook):
    return forward_ablate_sae(acts, sparse_autoencoder, None)[1]

def replacement_hook_ablate(acts, hook, ablate_feature_list):
    return forward_ablate_sae(acts, sparse_autoencoder, ablate_feature_list)[1]
    
def mean_ablate_hook(acts, hook):
    acts[:] = acts.mean([0, 1])
    return acts

def zero_ablate_hook(acts, hook):
    acts[:] = 0.
    return acts

def forward_ablate_sae(input_activations, sparse_autoencoder, ablate_feature_list=None):

    hidden_pre = einops.einsum(input_activations, sparse_autoencoder.W_enc, "... d_in, d_in d_sae -> ... d_sae") + sparse_autoencoder.b_enc
    
    feature_activations = torch.nn.functional.relu(hidden_pre)

    if ablate_feature_list is not None:
        n_features = feature_activations.shape[-1]
        ablate_flag = torch.zeros(n_features, dtype=torch.bool)
        ablate_flag[ablate_feature_list] = True
        feature_activations[:, :, ablate_flag] = 0
    
    output_activations = einops.einsum(feature_activations, sparse_autoencoder.W_dec, "... d_sae, d_sae d_in -> ... d_in") + sparse_autoencoder.b_dec
    
    return feature_activations, output_activations


def forward_offset_sae(input_activations, sparse_autoencoder, offset=0):

    hidden_pre = einops.einsum(input_activations, sparse_autoencoder.W_enc, "... d_in, d_in d_sae -> ... d_sae") + sparse_autoencoder.b_enc
    
    feature_activations = torch.nn.functional.relu(hidden_pre - offset)
    
    output_activations = einops.einsum(feature_activations, sparse_autoencoder.W_dec, "... d_sae, d_sae d_in -> ... d_in") + sparse_autoencoder.b_dec
    
    return feature_activations, output_activations


def create_lineplot_histogram(distribution, bins=20):
    vals, bin_edges = np.histogram(distribution, bins=bins)

    xvals = np.repeat(bin_edges, 2)
    yvals = np.repeat(vals, 2)
    yvals = np.concatenate(([0], yvals, [0]))

    return xvals, yvals


# def get_token_df_learned_activations(sparse_autoencoder, activation_store, model, n_batches=20, len_prefix=5):
#     cfg = sparse_autoencoder.cfg
#     l0_list = []
#     feature_activations_list = []
#     input_tokens_list = []
#     n_activations_sum = torch.zeros(cfg.d_in * cfg.expansion_factor).to(cfg.device)
    
#     with torch.no_grad():
#         for i in range(n_batches):
#             # Get batch of activations
#             input_activations, input_tokens = activation_store.next_batch()
        
#             # Forward pass
#             feature_activations, output_activations = sparse_autoencoder(input_activations)
#             n_new_activations = (feature_activations > 0).sum(dim=0)
            
#             n_activations_sum = n_activations_sum + n_new_activations
    
#             l0 = (feature_activations > 0).float().sum(-1).mean().item()
#             l0_list.append(l0)
#             feature_activations_list.append(feature_activations)
#             input_tokens_list.append(input_tokens)
    
    
#     total_inputs = n_batches * cfg.train_batch_size
#     sparsity = n_activations_sum / total_inputs

#     print("Concatenating learned activations")
#     token_df = make_token_df(model, torch.cat(input_tokens_list).reshape(-1, 128).to(int), len_prefix=len_prefix)
#     learned_activations = torch.cat(feature_activations_list)
#     print("Done")

#     return token_df, learned_activations, sparsity


def get_token_df_learned_activations(sparse_autoencoder, activation_store, model, n_batches=20, len_prefix=5):
    cfg = sparse_autoencoder.cfg
    l0_list = []
    feature_activations_list = []
    input_tokens_list = []
    n_activations_sum = torch.zeros(cfg.d_in * cfg.expansion_factor).to(cfg.device)
    
    with torch.no_grad():
        for i in range(n_batches):
            # Get batch of activations
            input_activations, input_tokens = activation_store.next_batch()
        
            # Forward pass
            feature_activations, output_activations = sparse_autoencoder(input_activations)
            n_new_activations = (feature_activations > 0).sum(dim=0)
            
            n_activations_sum = n_activations_sum + n_new_activations
    
            l0 = (feature_activations > 0).float().sum(-1).mean().item()
            l0_list.append(l0)
            feature_activations_list.append(feature_activations)
            input_tokens_list.append(input_tokens)
    
    
    total_inputs = n_batches * cfg.train_batch_size
    sparsity = n_activations_sum / total_inputs

    mean_l0 = torch.mean(l0_list)

    print("Concatenating learned activations")
    token_df = make_token_df(model, torch.cat(input_tokens_list).reshape(-1, 128).to(int), len_prefix=len_prefix)
    learned_activations = torch.cat(feature_activations_list)
    print("Done")

    return token_df, learned_activations, sparsity, mean_l0


def get_dfmax(token_df, learned_activations, feature_id):
    token_df["feature"] = utils.to_numpy(learned_activations[:, feature_id])
    df = token_df[['str_tokens','prefix', 'suffix',  'context', 'batch', 'pos', 'feature']]
    df = df.sort_values("feature", ascending=False)
    return df













