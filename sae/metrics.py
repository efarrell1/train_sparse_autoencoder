import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
import einops
import pandas as pd
import numpy as np


@torch.no_grad()
def compute_simple_metrics(target_activations, feature_activations, output_activations):
    """Computes metrics"""

    metrics = {}
    
    # L0
    l0 = (feature_activations > 0).float().sum(-1).mean()
    l0_std = (feature_activations > 0).float().sum(-1).std()
    metrics['l0'] = l0
    metrics['l0_std'] = l0_std

    # L1
    mean_activation = feature_activations.mean()
    std_activation = feature_activations.std()
    metrics['mean_activation'] = mean_activation
    metrics['std_activation'] = std_activation

    # Variance explained
    per_token_l2_loss = (output_activations - target_activations).pow(2).sum(dim=-1).squeeze()
    total_variance = (target_activations - target_activations.mean(0)).pow(2).sum(-1)
    explained_variance = 1 - per_token_l2_loss / total_variance
    unexplained_variance = 1 - explained_variance
    metrics['per_token_l2_loss'] = per_token_l2_loss.mean()
    metrics['total_variance'] = total_variance.mean()
    metrics['explained_variance'] = explained_variance.mean()
    metrics['unexplained_variance'] = unexplained_variance.mean()

    # MSE
    mse = ((output_activations - target_activations) ** 2).sum(dim=0).mean()
    metrics['mse'] = mse

    # L2 norm ratio
    metrics['l2_norm_ratio'] =  torch.norm(output_activations, p=2) / torch.norm(target_activations, p=2)
    
    # Old calculation method to compare with Joseph Bloom's models
    per_token_l2_loss = (output_activations - target_activations).pow(2).sum(dim=-1).squeeze()
    total_variance = target_activations.pow(2).sum(-1)
    metrics['old_explained_variance'] = (1 - per_token_l2_loss/total_variance).mean()

    for m, k in metrics.items():
        if isinstance(k, torch.Tensor):
            metrics[m] = k.item()

    return metrics

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists


@torch.no_grad()
def compute_metrics_post(sae_function, activation_store, model, save_learned_activations=True, n_batches=20, len_prefix=5, context_len=1024):

    # Computes post-hoc metrics for a given SAE, activation store and model
    
    metrics_list = []
    learned_activations = []
    input_tokens_list = []
    
    for i in tqdm(range(n_batches)):
        # Get batch of activations
        input_activations, input_tokens = activation_store.next_batch()
    
        # Forward pass
        try:
            try:
                feature_activations, output_activations = sae_function(input_activations)
            except ValueError:
                output_activations = sae_function(input_activations)
                feature_activations = sae_function.encode(input_activations)
        except:
            raise ValueError("SAE function must return a tuple of (feature_activations, output_activations) or just output_activations")
            

        metrics = compute_simple_metrics(input_activations, feature_activations, output_activations)
        metrics_list.append(metrics)

        # For sparsity calculation
        n_new_activations = (feature_activations > 0).sum(dim=0)
        if i == 0:
            n_activations_sum = torch.zeros(feature_activations.shape[1]).to(n_new_activations.device)
        n_activations_sum = n_activations_sum + n_new_activations

        # For token_df
        input_tokens_list.append(input_tokens.cpu())

        # For learned_activations
        if save_learned_activations:
            learned_activations.append(feature_activations.to(torch.float16).cpu())

    metrics = list_of_dicts_to_dict_of_lists(metrics_list)
    metrics = {k: torch.tensor(v).mean().item() for k, v in metrics.items()}

    # Calculate sparsity
    total_inputs = n_batches * input_activations.shape[0]
    sparsity = n_activations_sum / total_inputs
    metrics['sparsity'] = sparsity

    tokens = torch.cat(input_tokens_list).reshape(-1, context_len).to(int)
    print("tokens", tokens.shape, context_len)
    token_df = make_token_df(model, tokens, len_prefix=len_prefix)
    metrics['token_df'] = token_df

    if save_learned_activations:
        print("Concatenating learned activations")
        learned_activations = torch.cat(learned_activations).to(torch.float16)
        print("Done")
    else:
        learned_activations = None
    
    metrics['learned_activations'] = learned_activations
        
    return metrics


# This is a dumb function. I'll add the functionality to the above function when I have time
@torch.no_grad()
def compute_metrics_post_by_text(sae_function, text, model, hook_point, hook_point_layer, save_learned_activations=True, n_batches=20, len_prefix=5):

    metrics_list = []
    learned_activations = []
    input_tokens_list = []
    
    for i in range(n_batches):
        # Get batch of activations
        input_tokens = model.to_tokens(text)
        input_activations = model.run_with_cache(
                input_tokens, names_filter=hook_point, stop_at_layer=hook_point_layer + 1
            )[1][hook_point][0]
        
        # input_activations, input_tokens = activation_store.next_batch()
    
        # Forward pass
        feature_activations, output_activations = sae_function(input_activations)

        print(feature_activations.shape, output_activations.shape)

        metrics = compute_simple_metrics(input_activations, feature_activations, output_activations)
        metrics_list.append(metrics)

        # For sparsity calculation
        n_new_activations = (feature_activations > 0).sum(dim=0)
        if i == 0:
            n_activations_sum = torch.zeros(feature_activations.shape[1]).to(n_new_activations.device)
        n_activations_sum = n_activations_sum + n_new_activations

        # For token_df
        input_tokens_list.append(input_tokens)

        # For learned_activations
        if save_learned_activations:
            learned_activations.append(feature_activations.to(torch.float32))

    metrics = list_of_dicts_to_dict_of_lists(metrics_list)
    metrics = {k: torch.tensor(v).mean().item() for k, v in metrics.items()}

    # Calculate sparsity
    total_inputs = n_batches * input_activations.shape[0]
    sparsity = n_activations_sum / total_inputs
    metrics['sparsity'] = sparsity

    print(torch.cat(input_tokens_list).shape)
    token_df = make_token_df(model, torch.cat(input_tokens_list).to(int), len_prefix=len_prefix)
    metrics['token_df'] = token_df

    if save_learned_activations:
        print("Concatenating learned activations")
        learned_activations = torch.cat(learned_activations).to(torch.float32)
        print("Done")
    else:
        learned_activations = None
    
    metrics['learned_activations'] = learned_activations
        
    return metrics    


# More reptition, but I'll fix it later:
@torch.no_grad()
def compute_metrics_post_by_tokens(sae_function, tokens, model, hook_point, hook_point_layer, save_learned_activations=True, batch_size=1, len_prefix=5):

    metrics_list = []
    feature_activations_list = []
    input_tokens_list = []

    print(tokens.shape, batch_size)
    n_batches = (tokens.shape[0] // batch_size)
    print(n_batches)
    
    for i in range(n_batches):
        # Get batch of activations
        input_tokens = tokens[i * batch_size:min((i+1) * batch_size, len(tokens))]
        input_activations = model.run_with_cache(
                input_tokens, names_filter=hook_point, stop_at_layer=hook_point_layer + 1
            )[1][hook_point][0]
        
        # input_activations, input_tokens = activation_store.next_batch()
    
        # Forward pass
        feature_activations, output_activations = sae_function(input_activations)

        print(feature_activations.shape, output_activations.shape)

        metrics = compute_simple_metrics(input_activations, feature_activations, output_activations)
        metrics_list.append(metrics)

        # For sparsity calculation
        n_new_activations = (feature_activations > 0).sum(dim=0)
        if i == 0:
            n_activations_sum = torch.zeros(feature_activations.shape[1]).to(n_new_activations.device)
        n_activations_sum = n_activations_sum + n_new_activations

        # For token_df
        input_tokens_list.append(input_tokens)

        # For learned_activations
        if save_learned_activations:
            feature_activations_list.append(feature_activations)

    metrics = list_of_dicts_to_dict_of_lists(metrics_list)
    metrics = {k: torch.tensor(v).mean().item() for k, v in metrics.items()}

    # Calculate sparsity
    total_inputs = n_batches * input_activations.shape[0]
    sparsity = n_activations_sum / total_inputs
    metrics['sparsity'] = sparsity

    print(torch.cat(input_tokens_list).shape)
    token_df = make_token_df(model, torch.cat(input_tokens_list).to(int), len_prefix=len_prefix)
    metrics['token_df'] = token_df

    if save_learned_activations:
        print("Concatenating learned activations")
        learned_activations = torch.cat(feature_activations_list)
        print("Done")
    else:
        learned_activations = None
    
    metrics['learned_activations'] = learned_activations
        
    return metrics    


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



# @torch.no_grad()
# def get_token_df_learned_activations(sae_function, activation_store, model, save_learned_activations=True, n_batches=20, len_prefix=5):
#     l0_list = []
#     feature_activations_list = []
#     input_tokens_list = []
#     unexplained_variance_list = []
    
#     for i in range(n_batches):
#         # Get batch of activations
#         input_activations, input_tokens = activation_store.next_batch()
    
#         # Forward pass
#         feature_activations, output_activations = sae_function(input_activations)

#         # Save outputs
#         n_new_activations = (feature_activations > 0).sum(dim=0)
#         if i == 0:
#             n_activations_sum = torch.zeros(feature_activations.shape[1]).to(n_new_activations.device)
#         n_activations_sum = n_activations_sum + n_new_activations

#         l0 = (feature_activations > 0).float().sum(-1).mean()
#         l0_list.append(l0)
        
#         per_token_l2_loss = (output_activations - input_activations).pow(2).sum(dim=-1).squeeze()
#         total_variance = (input_activations - input_activations.mean(0)).pow(2).sum(-1)
#         unexplained_variance = (per_token_l2_loss / total_variance).mean()
#         unexplained_variance_list.append(unexplained_variance)
    
#         input_tokens_list.append(input_tokens)
#         if save_learned_activations:
#             feature_activations_list.append(feature_activations)

#     total_inputs = n_batches * input_activations.shape[0]
#     sparsity = n_activations_sum / total_inputs

#     mean_l0 = torch.tensor(l0_list).mean().item()
#     unexplained_variance = torch.tensor(unexplained_variance_list).mean().item()

#     token_df = make_token_df(model, torch.cat(input_tokens_list).reshape(-1, 128).to(int), len_prefix=len_prefix)

#     if save_learned_activations:
#         print("Concatenating learned activations")
#         learned_activations = torch.cat(feature_activations_list)
#         print("Done")
#     else:
#         learned_activations = None
        
#     return token_df, learned_activations, sparsity, mean_l0, unexplained_variance



def replacement_hook_ablate(ablate_feature_list, acts, hook):
    return forward_ablate_sae(acts, sparse_autoencoder, ablate_feature_list)[1]
    
def mean_ablate_hook(acts, hook):
    acts[:] = acts.mean([0, 1])
    return acts

def zero_ablate_hook(acts, hook):
    acts[:] = 0.
    return acts

    
@torch.no_grad()
def forward_offset_sae(sparse_autoencoder, offset, input_activations):
    """adds constant offset to hidden pre-RELU activations"""

    hidden_pre = einops.einsum(input_activations, sparse_autoencoder.W_enc, "... d_in, d_in d_sae -> ... d_sae") + sparse_autoencoder.b_enc
    
    feature_activations = torch.nn.functional.relu(hidden_pre - offset)
    
    output_activations = einops.einsum(feature_activations, sparse_autoencoder.W_dec, "... d_sae, d_sae d_in -> ... d_in") + sparse_autoencoder.b_dec
    
    return feature_activations, output_activations


@torch.no_grad()
def forward_topk_sae(sparse_autoencoder, topk, input_activations):
    """Selects top topk features by activation and sets the rest to 0"""
    hidden_pre = einops.einsum(input_activations, sparse_autoencoder.W_enc, "... d_in, d_in d_sae -> ... d_sae") + sparse_autoencoder.b_enc
    
    feature_activations = torch.nn.functional.relu(hidden_pre)
    vals, inds = torch.sort(feature_activations, descending=True, dim=1)
    discard_inds = (inds < topk)

    feature_activations[discard_inds] = 0

    output_activations = einops.einsum(feature_activations, sparse_autoencoder.W_dec, "... d_sae, d_sae d_in -> ... d_in") + sparse_autoencoder.b_dec
    
    return feature_activations, output_activations



@torch.no_grad()
def compute_recovered_loss(sae_function, activation_store, model, hook_point, head_index=None, n_batches=2):


    def replacement_hook(activations, hook):
        if hook_point.endswith(("hook_q", "hook_k", "hook_v")):
            new_activations = sae_function.forward(activations[:, :, head_index])[1].to(activations.dtype)
            activations[:, :, head_index] = new_activations
            return activations
        else:
            return sae_function(activations)[1].to(activations.dtype)

    loss_list = []
    for i in range(n_batches):
        tokens = activation_store.get_batch_tokenized_data()
        loss = model(tokens, return_type="loss")

        recons_loss = model.run_with_hooks(tokens, return_type="loss",
                                           fwd_hooks=[(hook_point, replacement_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss",
                                             fwd_hooks=[(hook_point, zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
    
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    scores = {"score": score,
              "loss": loss,
              "recons_loss": recons_loss,
              "zero_abl_loss": zero_abl_loss,
              "unrecovered_loss_frac": 1 - score,
              "frac_loss_added": (recons_loss - loss)/loss} 
    return scores



@torch.no_grad()
def compute_recovered_loss_transcoder(sae_function, activation_store, model, hook_point, hook_point_output, head_index=None, n_batches=2):


    # def replacement_hook(activations, hook):
    #     if hook_point.endswith(("hook_q", "hook_k", "hook_v")):
    #         new_activations = sae_function.forward(activations[:, :, head_index])[1].to(activations.dtype)
    #         activations[:, :, head_index] = new_activations
    #         return activations
    #     else:
    #         return sae_function(activations)[1].to(activations.dtype)

        

    loss_list = []
    for i in range(n_batches):
        tokens = activation_store.get_batch_tokenized_data()
        loss = model(tokens, return_type="loss")
        print("main base loss", loss)
        
        transcoder_outputs = None
        def calculate_transcoder_outputs(activations, hook):
            model.transcoder_outputs = sae_function(activations)[1]
            # print(model.transcoder_outputs.shape, model.transcoder_outputs[0][0][1])
            return activations

        def write_transcoder_outputs(activations, hook):
            # print(model.transcoder_outputs.shape, model.transcoder_outputs[0][0][1])
            return model.transcoder_outputs
            
        # old_mlp = model.blocks[hook_point_layer]
        
        # class TranscoderWrapper(torch.nn.Module):
        #     def __init__(self, transcoder):
        #         super().__init__()
        #         self.transcoder = transcoder
        #     def forward(self, x):
        #         return self.transcoder(x)[1]
                
        # model.blocks[hook_point_layer].mlp = TranscoderWrapper(sae_function)

        # recons_loss = model.run_with_hooks(tokens, return_type="loss")
        
        # model.blocks[hook_point_layer] = old_mlp
    
        loss = model(tokens, return_type="loss")
        print("main base loss", loss)

        recons_loss = model.run_with_hooks(tokens, return_type="loss",
                                           fwd_hooks=[(hook_point, calculate_transcoder_outputs), (hook_point_output, write_transcoder_outputs)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss",
                                             fwd_hooks=[(hook_point, zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
    
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    scores = {"score": score,
              "loss": loss,
              "recons_loss": recons_loss,
              "zero_abl_loss": zero_abl_loss,
              "unrecovered_loss_frac": 1 - score} 
    return scores



def model_store_from_sae(sae):
    model = HookedTransformer.from_pretrained(sae.cfg.model_name)
    model.to(sae.cfg.device)
    return model



### This is modified from Neel Nanda's code ###
def make_token_df(model, tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    prefix_list = []
    suffix_list = []
    print("tokens", tokens.shape)
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
### End token_df Neel Nanda's code ###
