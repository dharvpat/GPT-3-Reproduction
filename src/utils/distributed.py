import torch.distributed as dist

def setup_distributed():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())