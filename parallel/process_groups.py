import torch.distributed as dist

def init_model_parallel(tp_size=1, pp_size=1):
    world_size = dist.get_world_size()
    dp_size = world_size // (tp_size * pp_size)
    rank = dist.get_rank()

    tp_rank = rank % tp_size
    pp_rank = (rank // tp_size) % pp_size
    dp_rank = rank // (tp_size * pp_size)

    tp_group = None
    pp_group = None
    dp_group = None

    if tp_size > 1:
        for i in range(pp_size * dp_size):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = group

    if pp_size > 1:
        for i in range(tp_size * dp_size):
            ranks = [i + j * tp_size for j in range(pp_size)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                pp_group = group

    if dp_size > 1:
        for i in range(tp_size * pp_size):
            ranks = [i + j * (tp_size * pp_size) for j in range(dp_size)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                dp_group = group

    return {
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": dp_rank,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "tp_group": tp_group,
        "pp_group": pp_group,
        "dp_group": dp_group,
    }


_MPU = None

def set_model_parallel(mpu):
    global _MPU
    _MPU = mpu

def get_model_parallel():
    return _MPU
