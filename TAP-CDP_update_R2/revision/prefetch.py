"""Download data/weights once and report the environment, run as a script (not python -c) so that
repository files such as compression.py never shadow standard-library modules (Python 3.14+)."""
import argparse

import torch
import torchvision
from torchvision import datasets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cifar_root", default=None)
    p.add_argument("--weights", nargs="*", default=[], choices=["resnet18", "resnet50"])
    p.add_argument("--env", action="store_true", help="print torch/GPU information")
    p.add_argument("--has_gpu", action="store_true", help="print 1 if CUDA is available, else 0")
    p.add_argument("--gpu_test", action="store_true", help="run a convolution on every visible GPU (fails if the build lacks kernels for it)")
    a = p.parse_args()
    if a.has_gpu:
        print(int(torch.cuda.is_available()))
        return
    if a.env:
        print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(),
              [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    if a.gpu_test:
        if not torch.cuda.is_available():
            print("gpu_test: no CUDA GPU visible")
        for i in range(torch.cuda.device_count()):
            x = torch.randn(8, 64, 28, 28, device=f"cuda:{i}")
            y = torch.nn.functional.conv2d(x, torch.randn(64, 64, 3, 3, device=f"cuda:{i}"), padding=1)
            torch.cuda.synchronize(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"gpu_test: cuda:{i} {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]}) OK, output {tuple(y.shape)}")
    if a.cifar_root:
        datasets.CIFAR10(a.cifar_root, download=True)
        datasets.CIFAR10(a.cifar_root, train=False, download=True)
    for w in a.weights:
        getattr(torchvision.models, w)(weights="IMAGENET1K_V1")


if __name__ == "__main__":
    main()
