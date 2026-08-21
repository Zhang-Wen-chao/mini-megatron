import torch
import torch.nn as nn
from parallel.tensor_parallel import VocabParallelEmbedding

class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len):
        super().__init__()
        self.token_embedding = VocabParallelEmbedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(-1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos_emb = self.position_embedding(pos_ids)
        tok_emb = self.token_embedding(input_ids)
        return tok_emb + pos_emb
