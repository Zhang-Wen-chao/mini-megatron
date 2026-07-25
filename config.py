import torch

NUM_LAYERS = 12
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
FFN_HIDDEN_SIZE = 3072
MAX_SEQ_LEN = 512
VOCAB_SIZE = 50304

MICRO_BATCH_SIZE = 4
LEARNING_RATE = 6e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 10
MAX_TRAIN_STEPS = 100
MAX_GRAD_NORM = 1.0

def get_model_config():
    return {
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "ffn_hidden_size": FFN_HIDDEN_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
    }

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
