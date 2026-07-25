import contextlib
import torch
import torch.distributed as dist
import time
from parallel.data_parallel import allreduce_grads
from parallel.process_groups import get_model_parallel


def train_pipeline(embedding, decoder_layers, ln_f, lm_head, loss_fn,
                  optimizer, scheduler, data_iter, B, S, V, device,
                  num_microbatches, pp_rank, pp_size, rank,
                  hidden_size, tp_size=1, log_interval=10, dp_group=None,
                  amp_ctx=None):
    if amp_ctx is None:
        amp_ctx = contextlib.nullcontext()
    """Serial pipeline: each stage processes one micro-batch per iteration.
    Stage 0 does a single warmup forward; the first stage's backward lags
    the last stage's forward by `warmup` steps to overlap (limited).
    Not a true interleaved 1F1B.
    """
    losses = []
    warmup = pp_size - pp_rank - 1
    total_steps = num_microbatches

    parameters = list(optimizer.param_groups[0]['params'])

    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    for step in range(total_steps):
        tokens, labels = next(data_iter)
        tokens, labels = tokens.to(device), labels.to(device)
        loss_mask = torch.ones(B, S, dtype=torch.float32, device=device)

        if pp_rank == 0:
            with amp_ctx:
                x = embedding(tokens)
                for layer in decoder_layers:
                    x = layer(x)
            torch.cuda.synchronize()
            req = dist.batch_isend_irecv([dist.P2POp(dist.isend, x.detach().contiguous(), peer=rank + tp_size)])
            for r in req:
                r.wait()
            grad_out = torch.zeros_like(x)
            req = dist.batch_isend_irecv([dist.P2POp(dist.irecv, grad_out, peer=rank + tp_size)])
            for r in req:
                r.wait()
            torch.cuda.synchronize()
            if step >= warmup:
                x.backward(grad_out)

        elif pp_rank == pp_size - 1:
            x_buf = torch.zeros(B, S, hidden_size, device=device)
            req = dist.batch_isend_irecv([dist.P2POp(dist.irecv, x_buf, peer=rank - tp_size)])
            for r in req:
                r.wait()
            torch.cuda.synchronize()
            with amp_ctx:
                x = x_buf.detach().clone().requires_grad_(True)
                x_input = x
                for layer in decoder_layers:
                    x = layer(x)
                x = ln_f(x)
                logits = lm_head(x)
                loss = loss_fn(logits, labels, loss_mask)
            grad_x = torch.autograd.grad(loss, x_input, retain_graph=True, create_graph=False)[0]
            req = dist.batch_isend_irecv([dist.P2POp(dist.isend, grad_x.contiguous(), peer=rank - tp_size)])
            for r in req:
                r.wait()
            loss.backward()
            torch.cuda.synchronize()
            losses.append(loss.item())
            if rank == 0 and (step + 1) % log_interval == 0 and step >= warmup:
                print(f"step {step+1-warmup:4d} | loss {loss.item():.4f}")

        if step == total_steps - 1:
            if dp_group is not None:
                allreduce_grads(parameters, dp_group)
            optimizer.step()
            scheduler.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if pp_rank == pp_size - 1 and losses:
        mpu = get_model_parallel()
        tp_rank = mpu["tp_rank"] if mpu else 0
        if tp_rank == 0:
            throughput = B * S * num_microbatches / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            amp_str = "BF16" if amp_ctx.__class__.__name__ != "nullcontext" else "FP32"
            print(f"\n{'='*60}")
            print(f"Mini-Megatron PP Results (AMP={amp_str})")
            print(f"{'='*60}")
            print(f"PP={pp_size}  |  Steps: {num_microbatches}")
            print(f"Throughput:      {throughput:,.0f} tok/s")
            print(f"Peak memory:     {peak_mem:.2f} GB/GPU")
            print(f"Final loss:      {losses[-1]:.4f}")
            print(f"{'='*60}\n")

    return elapsed
