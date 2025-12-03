import argparse
import sys

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.645)
    parser.add_argument("--dof", type=float, default=5)
    parser.add_argument("--samples", type=int, default=2**18)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    samples = torch.distributions.StudentT(args.dof).sample((args.samples,))
    idx = samples.div(args.delta).round().int().clamp(-128, 127)
    count = torch.bincount(idx + 128, minlength=256)

    prob = count.div(count.sum())
    entropy = prob.float().log2().mul(prob).nan_to_num(0).sum().neg().item()
    print(f"Entropy: {entropy:.4f} bits", file=sys.stderr)

    sys.stdout.buffer.write(count.numpy().tobytes())
