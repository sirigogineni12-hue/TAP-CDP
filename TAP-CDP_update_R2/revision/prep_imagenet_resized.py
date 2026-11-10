"""One-time copy of the ImageNet training set with the shorter side resized to 256 pixels (JPEG quality 95),
so that 8 CPUs can feed training. Images whose shorter side is already <= 256 are copied unchanged.
The validation set is not resized: <out>/val links to the original validation folder.
  python revision/prep_imagenet_resized.py --src results/data/imagenet --out results/data/imagenet256 --workers 8
"""
import argparse
import os
import shutil
from multiprocessing import Pool

from PIL import Image

SIZE = 256


def complete_jpeg(path):
    """True if the file exists and ends with the JPEG end-of-image marker. Files cut off by a full disk
    (empty or truncated) fail this test and are converted again."""
    try:
        if os.path.getsize(path) < 4:
            return False
        with open(path, "rb") as f:
            f.seek(-2, os.SEEK_END)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


def convert(job):
    src, dst = job
    if complete_jpeg(dst):
        return 0
    try:
        with Image.open(src) as img:
            w, h = img.size
            if min(w, h) <= SIZE:
                shutil.copyfile(src, dst)
                return 1
            s = SIZE / min(w, h)
            img.convert("RGB").resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR) \
                .save(dst, "JPEG", quality=95)
        return 1
    except Exception as e:  # keep going; report at the end, and remove any partial file
        print(f"failed: {src}: {e}", flush=True)
        if os.path.exists(dst):
            os.remove(dst)
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="root with train/ and val/")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    train = os.path.join(a.src, "train")
    jobs = []
    for wnid in sorted(os.listdir(train)):
        os.makedirs(os.path.join(a.out, "train", wnid), exist_ok=True)
        for f in os.listdir(os.path.join(train, wnid)):
            jobs.append((os.path.join(train, wnid, f), os.path.join(a.out, "train", wnid, f)))
    print(f"{len(jobs)} training images", flush=True)
    done = failed = 0
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(convert, jobs, chunksize=256), 1):
            done += r == 1
            failed += r == -1
            if i % 100000 == 0:
                print(f"{i} / {len(jobs)}", flush=True)
    val_link = os.path.join(a.out, "val")
    if not os.path.exists(val_link):
        os.symlink(os.path.abspath(os.path.join(a.src, "val")), val_link)
    print(f"converted {done}, failed {failed}, already complete {len(jobs) - done - failed}, total {len(jobs)}")
    if failed:
        raise SystemExit(f"{failed} images failed; fix the cause (e.g. disk space) and run again")


if __name__ == "__main__":
    main()
