"""Checks for the corrected block_dct. Run: python revision/test_block_dct.py"""
import torch

from common import block_dct, repo_block_dct


def main():
    torch.manual_seed(0)
    # 1. Identical to the repository's function where that one runs (H = W = 512).
    x = torch.randn(2, 1, 512, 512)
    diff = (block_dct(x) - repo_block_dct(x)).abs().max().item()
    print(f"max |corrected - original| at 512x512: {diff:.2e}")
    assert diff < 1e-4

    # 2. The original fails at the paper's input sizes; the corrected one works.
    for size in (32, 224):
        try:
            repo_block_dct(torch.zeros(1, 1, size, size))
            print(f"original at {size}x{size}: runs")
        except RuntimeError:
            print(f"original at {size}x{size}: RuntimeError (as expected)")
        print(f"corrected at {size}x{size}: output {tuple(block_dct(torch.zeros(1, 1, size, size)).shape)}")

    # 3. Orthonormal transform: energy is preserved (Parseval).
    x = torch.randn(3, 1, 224, 224, dtype=torch.float64)
    e_in, e_out = (x ** 2).sum().item(), (block_dct(x) ** 2).sum().item()
    print(f"energy in {e_in:.4f} / out {e_out:.4f}")
    assert abs(e_in - e_out) / e_in < 1e-10

    # 4. Band layout: a constant block puts all energy in the DC band (index 0);
    #    a pure horizontal cosine of frequency 3 lands in band 0*8 + 3.
    const = torch.ones(1, 1, 8, 8, dtype=torch.float64)
    c = block_dct(const).view(64)
    assert abs(c[0].item() - 8.0) < 1e-9 and c[1:].abs().max().item() < 1e-9
    n = torch.arange(8, dtype=torch.float64)
    horiz = torch.cos(torch.pi * (2 * n + 1) * 3 / 16).repeat(8, 1).view(1, 1, 8, 8)
    c = block_dct(horiz).view(64)
    assert c.abs().argmax().item() == 3
    print("all block_dct checks passed")


if __name__ == "__main__":
    main()
