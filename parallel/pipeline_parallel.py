import contextlib
import torch
import torch.distributed as dist
import time
from parallel.data_parallel import allreduce_grads
from parallel.process_groups import get_model_parallel


def build_1f1b_schedule(pp_rank, pp_size, num_microbatches):
    """1F1B op order for one stage: ('F'|'B', microbatch index) pairs.

    Mirrors Megatron-LM's non-interleaved schedule: a stage runs
    `pp - rank - 1` warmup forwards, then alternates forward (next
    micro-batch) / backward (oldest outstanding), then drains with the
    remaining backwards. Bubble drops from (pp-1)/pp (serial lockstep)
    to (pp-1)/(2m+pp-1).
    """
    warmup = min(pp_size - pp_rank - 1, num_microbatches)
    ops = []
    fwd, bwd = 0, 0
    for _ in range(warmup):
        ops.append(("F", fwd))
        fwd += 1
    while fwd < num_microbatches:
        ops.append(("F", fwd))
        fwd += 1
        ops.append(("B", bwd))
        bwd += 1
    while bwd < num_microbatches:
        ops.append(("B", bwd))
        bwd += 1
    return ops


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


def train_pipeline_1f1b(embedding, decoder_layers, ln_f, lm_head, loss_fn,
                        optimizer, scheduler, data_iter, B, S, V, device,
                        num_microbatches, pp_rank, pp_size, rank,
                        hidden_size, tp_size=1, log_interval=10, dp_group=None,
                        amp_ctx=None):
    """1F1B pipeline schedule: forward/backward interleaved per stage.

    Backward of micro-batch j fills the bubble left by forward of j+1, so
    idle time drops from (pp-1)/pp (serial lockstep) to (pp-1)/(2m+pp-1).
    Forward recvs are posted one op early (look-ahead) so the transfer
    overlaps the previous op's compute; sends are fire-and-forget and
    waited a couple of ops later. Works for any pp >= 2.
    """
    if amp_ctx is None:
        amp_ctx = contextlib.nullcontext()
    m = num_microbatches
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1
    prev_peer = rank - tp_size
    next_peer = rank + tp_size
    loss_mask = torch.ones(B, S, dtype=torch.float32, device=device)

    fwd_input = {}   # mb -> input to this stage's layers (graph root)
    fwd_out = {}     # mb -> forward output (non-leaf, for the backward pass)
    grad_in = {}     # mb -> received gradient for its backward
    loss_list = []   # mb -> loss tensor (last stage only, in F order)
    ops = build_1f1b_schedule(pp_rank, pp_size, m)

    send_reqs = []   # outstanding (request, tensor) so sends stay alive
    fwd_req = None   # pending irecv of the next forward's input
    bwd_req = None   # pending irecv of the next backward's gradient

    def wait_sends():
        while len(send_reqs) > 1:
            req, _ = send_reqs.pop(0)
            req.wait()

    def post_fwd_recv(mb):
        buf = torch.zeros(B, S, hidden_size, device=device).requires_grad_(True)
        fwd_input[mb] = buf
        return dist.irecv(buf, src=prev_peer)

    def post_bwd_recv(mb):
        buf = torch.zeros(B, S, hidden_size, device=device)
        grad_in[mb] = buf
        return dist.irecv(buf, src=next_peer)

    def run_forward(mb):
        if is_first:
            tokens, _ = next(data_iter)
            with amp_ctx:
                x = embedding(tokens.to(device))
                for layer in decoder_layers:
                    x = layer(x)
            fwd_input[mb] = x
        else:
            x = fwd_input[mb]
            if is_last:
                _, labels = next(data_iter)
            with amp_ctx:
                for layer in decoder_layers:
                    x = layer(x)
                if is_last:
                    x = ln_f(x)
                    logits = lm_head(x)
                    loss = loss_fn(logits, labels.to(device), loss_mask)
                    loss_list.append(loss)
        if not is_last:
            fwd_out[mb] = x
            send_reqs.append((dist.isend(x.contiguous(), dst=next_peer), x))

    def run_backward(mb):
        x = fwd_input[mb]
        if is_first:
            x.backward(grad_in[mb])
        elif is_last:
            loss_list[mb].backward()
        else:
            fwd_out[mb].backward(grad_in[mb])
        if not is_first:
            send_reqs.append((dist.isend(x.grad.contiguous(), dst=prev_peer), x.grad))
        wait_sends()

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    parameters = list(optimizer.param_groups[0]["params"])
    optimizer.zero_grad()
    sync()
    start = time.perf_counter()

    for i, (kind, mb) in enumerate(ops):
        if kind == "F":
            if not is_first:
                if fwd_req is None:
                    fwd_req = post_fwd_recv(mb)
                fwd_req.wait()
                fwd_req = None
            run_forward(mb)
        else:
            if not is_last:
                if bwd_req is None:
                    bwd_req = post_bwd_recv(mb)
                bwd_req.wait()
                bwd_req = None
            run_backward(mb)
        if is_last and rank == (pp_size - 1) * tp_size and kind == "B" \
                and (mb + 1) % log_interval == 0:
            print(f"step {mb+1:4d} | loss {loss_list[mb].item():.4f}")
        # Look-ahead: post the next op's recv so the transfer overlaps
        # this op's compute.
        if i + 1 < len(ops):
            nk, nmb = ops[i + 1]
            if nk == "F" and not is_first:
                fwd_req = post_fwd_recv(nmb)
            elif nk == "B" and not is_last:
                bwd_req = post_bwd_recv(nmb)
        if kind == "B":
            fwd_input.pop(mb, None)
            fwd_out.pop(mb, None)
            grad_in.pop(mb, None)

    for req, _ in send_reqs:
        req.wait()

    if dp_group is not None:
        allreduce_grads(parameters, dp_group)
    optimizer.step()
    scheduler.step()

    sync()
    elapsed = time.perf_counter() - start

    if is_last and loss_list:
        mpu = get_model_parallel()
        tp_rank = mpu["tp_rank"] if mpu else 0
        if tp_rank == 0:
            throughput = B * S * m / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            amp_str = "BF16" if amp_ctx.__class__.__name__ != "nullcontext" else "FP32"
            print(f"\n{'='*60}")
            print(f"Mini-Megatron PP Results (1F1B, AMP={amp_str})")
            print(f"{'='*60}")
            print(f"PP={pp_size}  |  Micro-batches: {m}")
            print(f"Throughput:      {throughput:,.0f} tok/s")
            print(f"Peak memory:     {peak_mem:.2f} GB/GPU")
            print(f"Final loss:      {loss_list[-1].item():.4f}")
            print(f"{'='*60}\n")

    return elapsed
