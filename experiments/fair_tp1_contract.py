"""Shared TP=1 GPT contract used by the fair mini/Megatron-Core experiment."""
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from model.embedding import Embedding
from model.transformer import Decoder, GPT
from parallel.tensor_parallel import ColumnParallelLinear

CONTRACT = (
    "12L-768H-12head-3072FFN, learned absolute position, pre-LN, GELU, "
    "no dropout, bias-free linears, causal next-token cross entropy"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mini(device):
    model_config = cfg.get_model_config()
    embedding = Embedding(model_config["vocab_size"], model_config["hidden_size"], model_config["max_seq_len"])
    decoder = Decoder(model_config["hidden_size"], model_config["num_attention_heads"],
                      model_config["ffn_hidden_size"], model_config["num_layers"])
    model = GPT(embedding, decoder, torch.nn.LayerNorm(model_config["hidden_size"]),
                ColumnParallelLinear(model_config["hidden_size"], model_config["vocab_size"],
                                     bias=False, gather_output=True)).to(device)
    return model, model_config


def parameter_mappings(model_config):
    pairs = [
        ("embedding.token_embedding.weight", "embedding.word_embeddings.weight"),
        ("embedding.position_embedding.weight", "embedding.position_embeddings.weight"),
        ("ln_f.weight", "decoder.final_layernorm.weight"),
        ("ln_f.bias", "decoder.final_layernorm.bias"),
        ("lm_head.weight", "output_layer.weight"),
    ]
    for index in range(model_config["num_layers"]):
        mini, mcore = f"decoder.layers.{index}", f"decoder.layers.{index}"
        pairs.extend([
            (f"{mini}.ln1.weight", f"{mcore}.input_layernorm.weight"),
            (f"{mini}.ln1.bias", f"{mcore}.input_layernorm.bias"),
            (f"{mini}.attn.qkv.weight", f"{mcore}.self_attention.linear_qkv.weight"),
            (f"{mini}.attn.o.weight", f"{mcore}.self_attention.linear_proj.weight"),
            (f"{mini}.ln2.weight", f"{mcore}.pre_mlp_layernorm.weight"),
            (f"{mini}.ln2.bias", f"{mcore}.pre_mlp_layernorm.bias"),
            (f"{mini}.mlp.fc1.weight", f"{mcore}.mlp.linear_fc1.weight"),
            (f"{mini}.mlp.fc2.weight", f"{mcore}.mlp.linear_fc2.weight"),
        ])
    return pairs


def is_qkv_pair(mini_name, mcore_name):
    return mini_name.endswith(".attn.qkv.weight") and mcore_name.endswith(".self_attention.linear_qkv.weight")


def qkv_mini_to_mcore(tensor, model_config):
    """[all Q | all K | all V] rows -> MCore's [Q_i | K_i | V_i] layout."""
    heads = model_config["num_attention_heads"]
    head_dim = model_config["hidden_size"] // heads
    if tensor.dim() != 2:
        raise ValueError("QKV conversion expects a 2-D weight matrix")
    return tensor.view(3, heads, head_dim, tensor.shape[1]).permute(1, 0, 2, 3).reshape_as(tensor)


def qkv_mcore_to_mini(tensor, model_config):
    heads = model_config["num_attention_heads"]
    head_dim = model_config["hidden_size"] // heads
    if tensor.dim() != 2:
        raise ValueError("QKV conversion expects a 2-D weight matrix")
    return tensor.view(heads, 3, head_dim, tensor.shape[1]).permute(1, 0, 2, 3).reshape_as(tensor)


def mcore_tensor_in_mini_layout(tensor, mini_name, mcore_name, model_config):
    return qkv_mcore_to_mini(tensor, model_config) if is_qkv_pair(mini_name, mcore_name) else tensor


def copy_mini_to_mcore(mini, mcore, model_config):
    mini_params, mcore_params = dict(mini.named_parameters()), dict(mcore.named_parameters())
    pairs = parameter_mappings(model_config)
    unmapped_mini = sorted(set(mini_params) - {left for left, _ in pairs})
    unmapped_mcore = sorted(set(mcore_params) - {right for _, right in pairs})
    if unmapped_mini or unmapped_mcore:
        raise RuntimeError("parameter mapping is incomplete: mini=" + repr(unmapped_mini) + " mcore=" + repr(unmapped_mcore))
    with torch.no_grad():
        for mini_name, mcore_name in pairs:
            if mini_params[mini_name].shape != mcore_params[mcore_name].shape:
                raise RuntimeError("shape mismatch: " + mini_name + " vs " + mcore_name)
            source = qkv_mini_to_mcore(mini_params[mini_name], model_config) if is_qkv_pair(mini_name, mcore_name) else mini_params[mini_name]
            mcore_params[mcore_name].copy_(source)
    return pairs
