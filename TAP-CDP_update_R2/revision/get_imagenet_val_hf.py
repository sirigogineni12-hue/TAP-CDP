"""Build results/data/imagenet/{train,val} for the ImageNet experiments. Run on the LOGIN node (needs internet).

* train/ is a symbolic link to the shared training set (one folder per class, 1,281,167 images).
* val/ is created from the validation split of the Hugging Face dataset ILSVRC/imagenet-1k (50,000 original
  JPEG files, about 6.4 GB): the parquet shards are downloaded and every image is written unchanged to
  val/<wnid>/. Label i is the i-th class in sorted WordNet-ID order, the standard ILSVRC-2012 order used by
  torchvision; the training folders give the WordNet IDs.
Before running: accept the dataset terms on https://huggingface.co/datasets/ILSVRC/imagenet-1k with your
Hugging Face account and log in on the login node (hf auth login).
  python revision/get_imagenet_val_hf.py --train /janaki/common/Datasets/Imagenet_1k_2012 --root results/data/imagenet
Check afterwards with job8 (torchvision ResNet-50 must give about 76.1% top-1).
"""
import argparse
import glob
import os

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="shared training set (one folder per class)")
    p.add_argument("--root", required=True, help="output root: creates train (link) and val")
    p.add_argument("--cache", default=None, help="where to keep the parquet shards (default: <root>_hf)")
    a = p.parse_args()

    os.makedirs(a.root, exist_ok=True)
    link = os.path.join(a.root, "train")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(a.train), link)
    wnids = sorted(d for d in os.listdir(a.train) if os.path.isdir(os.path.join(a.train, d)))
    assert len(wnids) == 1000, f"expected 1000 class folders in {a.train}, found {len(wnids)}"

    cache = a.cache or a.root.rstrip("/") + "_hf"
    snapshot_download("ILSVRC/imagenet-1k", repo_type="dataset", allow_patterns=["data/validation-*"],
                      local_dir=cache)
    shards = sorted(glob.glob(os.path.join(cache, "data", "validation-*.parquet")))
    assert shards, "no validation shards downloaded"
    val = os.path.join(a.root, "val")
    n = 0
    for s in shards:
        t = pq.read_table(s)
        for img, label in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            d = os.path.join(val, wnids[label])
            os.makedirs(d, exist_ok=True)
            name = os.path.basename(img.get("path") or "") or f"val_{n:08d}.JPEG"
            if not name.upper().endswith((".JPEG", ".JPG")):
                name += ".JPEG"
            with open(os.path.join(d, name), "wb") as f:
                f.write(img["bytes"])
            n += 1
        print(f"{os.path.basename(s)}: {n} images written", flush=True)
    counts = [len(os.listdir(os.path.join(val, w))) for w in wnids if os.path.isdir(os.path.join(val, w))]
    print(f"val: {n} images in {len(counts)} classes (min {min(counts)}, max {max(counts)} per class)")
    print(f"parquet shards kept in {cache}; delete them after job8 confirms the labels")


if __name__ == "__main__":
    main()
