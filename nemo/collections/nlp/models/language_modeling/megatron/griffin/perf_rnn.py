import argparse
from functools import partial
import torch
import numpy as np
from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_module import (
    RGLRU,
    Conv1D,
    RecurrentLayer,
    RecurrentLayerSubmodules,
)
#import os
#os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS') + ' --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=/scratch/my_work/nsys/jax/gemma/hlo_out/rglru_b2_s4096_h10_d256_auto'
# nsys_cmd="nsys profile -s none -t nvtx,cuda -o /home/ataghibakhsh/nsys_results --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"


def train_step(model, x, segment_pos, y_grad):
    # Ensure all inputs are tensors and have requires_grad if necessary
    # x.requires_grad_(True)
    # segment_pos.requires_grad_(True)  # Assuming we need gradients w.r.t. segment_pos

    # Forward pass
    out, _ = model(x, segment_pos, None)

    # Backward pass
    # out.backward(y_grad)  
    # # Extract gradients
    x_grad = None# x.grad

    # Return the output, gradients of the parameters, and gradients of the input
    return out, x_grad


def main():
    parser = argparse.ArgumentParser(description='T5X Softmax Unit Test')
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=2048)
    parser.add_argument("--num_heads", dest="num_heads", type=int, default=10)
    parser.add_argument("--head_dim", dest="head_dim", type=int, default=256)
    parser.add_argument("--expand", dest="expand", type=int, default=1)
    parser.add_argument("--scan_type", dest="scan_type", type=str, default="auto")
    parser.add_argument("--profile", action="store_true", default=True)
    args = parser.parse_args()

    args.num_heads = args.num_heads * args.expand
    args.hidden_size = args.num_heads * args.head_dim

    dtype = torch.bfloat16
    x = torch.rand((args.batch_size, args.seq_len, args.hidden_size), dtype=dtype).cuda()
    segment_pos = torch.tile(torch.arange(args.seq_len), (args.batch_size, 1)).cuda()
    y_grad = torch.rand((args.batch_size, args.seq_len, args.hidden_size), dtype=torch.float32).cuda()
    # h_grad = torch.rand((args.batch_size, args.hidden_size), dtype=torch.float32).cuda()

    model = RGLRU(args.hidden_size, args.num_heads).cuda()
    model = model.to(dtype=torch.bfloat16)
    
    # params = model.init(x, segment_pos)

    # jitted_train_step = jax.jit(train_step, static_argnums=0)

    if args.profile:
        for i in range(100):

            torch.cuda.nvtx.range_push("rglru-block")
            out, x_grad = train_step(model, x, segment_pos, y_grad)
            torch.cuda.nvtx.range_pop()
    else:
        for i in range(100):
            out, x_grad = train_step(model, x, segment_pos, y_grad)

    return out, x_grad

if __name__ == "__main__":
    main()
