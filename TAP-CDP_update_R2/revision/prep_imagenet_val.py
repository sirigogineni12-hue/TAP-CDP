"""Sort the 50,000 ImageNet validation images of the Kaggle download into one folder per class.

LOC_val_solution.csv has rows "ImageId,PredictionString" where the first token of PredictionString is the
class (WordNet ID) of the image. Images are moved from --src (flat folder) to --dst/<wnid>/.
"""
import argparse
import csv
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solution", required=True, help="LOC_val_solution.csv")
    p.add_argument("--src", required=True, help="flat folder with ILSVRC2012_val_*.JPEG")
    p.add_argument("--dst", required=True, help="output folder (one subfolder per class)")
    a = p.parse_args()
    moved = 0
    with open(a.solution) as f:
        for row in csv.DictReader(f):
            wnid = row["PredictionString"].split()[0]
            src = os.path.join(a.src, row["ImageId"] + ".JPEG")
            os.makedirs(os.path.join(a.dst, wnid), exist_ok=True)
            if os.path.exists(src):
                shutil.move(src, os.path.join(a.dst, wnid, row["ImageId"] + ".JPEG"))
                moved += 1
    classes = [d for d in os.listdir(a.dst) if os.path.isdir(os.path.join(a.dst, d))]
    total = sum(len(os.listdir(os.path.join(a.dst, d))) for d in classes)
    print(f"moved {moved} images; {len(classes)} classes, {total} validation images in {a.dst}")


if __name__ == "__main__":
    main()
