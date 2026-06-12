#!/bin/bash

# datasets_setup.sh - Script to prepare data directories for TAP-CDP

DATA_DIR="./data"
mkdir -p $DATA_DIR

echo ">>> Preparing CIFAR-10..."
# CIFAR-10 is automatically handled by torchvision in dataset.py, 
# but we can pre-create the directory.
mkdir -p $DATA_DIR/cifar10

echo ">>> Preparing ImageNet (ILSVRC-2012)..."
mkdir -p $DATA_DIR/imagenet/train
mkdir -p $DATA_DIR/imagenet/val

echo "------------------------------------------------------------------------"
echo "NOTE: ImageNet (ILSVRC-2012) cannot be downloaded automatically due to"
echo "size and licensing restrictions. Please follow these steps:"
echo ""
echo "1. Download 'ILSVRC2012_img_train.tar' and 'ILSVRC2012_img_val.tar'"
echo "   from https://image-net.org/"
echo ""
echo "2. Move them to $DATA_DIR/imagenet/"
echo ""
echo "3. Extract training data:"
echo "   mkdir -p $DATA_DIR/imagenet/train && tar -xvf ILSVRC2012_img_train.tar -C $DATA_DIR/imagenet/train"
echo ""
echo "4. Extract validation data:"
echo "   mkdir -p $DATA_DIR/imagenet/val && tar -xvf ILSVRC2012_img_val.tar -C $DATA_DIR/imagenet/val"
echo ""
echo "5. For ImageNet training, ensure you run the extraction script to"
echo "   organize subfolders (e.g., using the standard PyTorch ImageNet script)."
echo "------------------------------------------------------------------------"

echo "Setup complete. Check $DATA_DIR for your dataset structures."
