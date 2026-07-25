"""Tests for model components: embedding, transformer, loss.

These tests do not require GPU; they run on CPU.
"""
import torch
import config as cfg
from model.embedding import Embedding
from model.transformer import Attention, MLP, DecoderLayer, Decoder, GPT
from model.loss import CrossEntropyLoss


def test_embedding_forward_shape():
    """Embedding output should be [B, S, H]."""
    embed = Embedding(vocab_size=100, hidden_size=16, max_seq_len=8)
    x = torch.randint(0, 100, (2, 8))
    out = embed(x)
    assert out.shape == (2, 8, 16)


def test_embedding_position_dependency():
    """Different positions should produce different outputs."""
    embed = Embedding(vocab_size=10, hidden_size=4, max_seq_len=4)
    tok = torch.tensor([[0, 0, 0, 0]])
    out = embed(tok)
    # Same token, different position → different output
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_attention_forward_shape():
    """Attention output should preserve [B, S, H]."""
    torch.manual_seed(0)
    attn = Attention(hidden_size=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    out = attn(x)
    assert out.shape == (2, 8, 16)


def test_mlp_forward_shape():
    torch.manual_seed(0)
    mlp = MLP(hidden_size=16, ffn_hidden_size=64)
    x = torch.randn(2, 8, 16)
    out = mlp(x)
    assert out.shape == (2, 8, 16)


def test_decoder_layer_forward_shape():
    torch.manual_seed(0)
    layer = DecoderLayer(hidden_size=16, num_heads=4, ffn_hidden_size=64)
    x = torch.randn(2, 8, 16)
    out = layer(x)
    assert out.shape == (2, 8, 16)


def test_decoder_stack():
    """Decoder should stack N layers."""
    torch.manual_seed(0)
    dec = Decoder(hidden_size=16, num_heads=4, ffn_hidden_size=64, num_layers=6)
    assert len(dec.layers) == 6
    x = torch.randn(2, 8, 16)
    out = dec(x)
    assert out.shape == (2, 8, 16)


def test_gpt_forward_with_and_without_loss():
    """GPT.forward should handle (input_ids,) and (input_ids, labels, loss_mask) cases."""
    torch.manual_seed(0)
    HS, NH, FFN, NL, V = 16, 4, 64, 2, 32
    embed = Embedding(V, HS, max_seq_len=8)
    decoder = Decoder(HS, NH, FFN, NL)
    ln_f = torch.nn.LayerNorm(HS)
    lm_head = torch.nn.Linear(HS, V, bias=False)
    loss_fn = CrossEntropyLoss()
    model = GPT(embed, decoder, ln_f, lm_head, loss_fn)

    x = torch.randint(0, V, (2, 8))

    # Without labels: returns logits only
    logits = model(x)
    assert logits.shape == (2, 8, V)

    # With labels + loss_mask: returns (logits, loss)
    loss_mask = torch.ones(2, 8)
    logits, loss = model(x, labels=x, loss_mask=loss_mask)
    assert logits.shape == (2, 8, V)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_cross_entropy_loss_uniform_input():
    """On uniform random predictions, loss should be ~ln(vocab_size)."""
    loss_fn = CrossEntropyLoss()
    B, S, V = 2, 4, 100
    # logits = 0 means uniform distribution → loss should be ln(V) ≈ 4.605
    logits = torch.zeros(B * S, V)
    labels = torch.randint(0, V, (B * S,))
    loss_mask = torch.ones(B * S)
    loss = loss_fn(logits, labels, loss_mask)
    expected = torch.log(torch.tensor(float(V))).item()
    assert abs(loss.item() - expected) < 0.01


def test_cross_entropy_loss_perfect_prediction():
    """On perfect predictions, loss should be ~0."""
    loss_fn = CrossEntropyLoss()
    B, S, V = 1, 1, 5
    logits = torch.full((B * S, V), -10.0)
    logits[0, 2] = 10.0  # predict class 2 with high confidence
    labels = torch.tensor([2])
    loss_mask = torch.ones(B * S)
    loss = loss_fn(logits, labels, loss_mask)
    assert loss.item() < 0.01
