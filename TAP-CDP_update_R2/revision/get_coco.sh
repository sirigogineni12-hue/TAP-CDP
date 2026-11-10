#!/bin/bash
# Run ONCE on the login node (it needs internet): downloads COCO val2017 (5,000 photos, about 0.8 GB) into
# results/data/coco/val2017. Used only as real JPEG input for latency/energy timing when ImageNet is not
# available (timing needs real photos, not labels). Images: cocodataset.org, Creative Commons licences.
#   bash revision/get_coco.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
if [ -d "$LATENCY_IMAGES" ] && [ "$(ls "$LATENCY_IMAGES" | wc -l)" -ge 5000 ]; then
  echo "COCO val2017 already in $LATENCY_IMAGES"; exit 0
fi
mkdir -p "$(dirname "$LATENCY_IMAGES")"
cd "$(dirname "$LATENCY_IMAGES")"
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && rm val2017.zip
echo "COCO val2017: $(ls val2017 | wc -l) images in $(pwd)/val2017"
