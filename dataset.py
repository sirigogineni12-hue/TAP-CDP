# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import io
import os

class JpegCompressedDataset(Dataset):
    """
    A wrapper for a dataset to apply JPEG compression on the fly.
    """
    def __init__(self, dataset, quality=50):
        self.dataset = dataset
        self.quality = quality
        self.transform = None

    def __getitem__(self, index):
        img, target = self.dataset[index]

        # Convert to PIL Image if it's a tensor
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Compress and decompress
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=self.quality)
        img = Image.open(buffer)

        # Apply the transformations
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dataset)

def get_data_loaders(config):
    """
    Creates and returns data loaders for the specified dataset.
    """
    dataset_name = config['dataset']['name']
    root = config['dataset']['root']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    spatial_res = config['dataset'].get('spatial_res', None)

    if dataset_name == 'CIFAR10':
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        test_transform_list = []

        if spatial_res:
            train_transform_list.append(transforms.Resize(spatial_res))
            test_transform_list.append(transforms.Resize(spatial_res))

        train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == 'ImageNet':
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, 'val')

        train_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]
        test_transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

        if spatial_res:
            train_transform_list.append(transforms.Resize(spatial_res))
            test_transform_list.append(transforms.Resize(spatial_res))

        train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Handle JPEG compression if specified
    if 'jpeg_quality' in config and config['jpeg_quality'] is not None:
        quality = config['jpeg_quality']
        # We need a version of the dataset without any transforms for JPEG compression
        if dataset_name == 'CIFAR10':
            train_dataset_unnormalized = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
            test_dataset_unnormalized = datasets.CIFAR10(root=root, train=False, download=True)
        
        train_dataset = JpegCompressedDataset(train_dataset_unnormalized, quality=quality)
        train_dataset.transform = train_transform # Re-apply full transform after compression
        
        test_dataset = JpegCompressedDataset(test_dataset_unnormalized, quality=quality)
        test_dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_classes

if __name__ == '__main__':
    # Example usage
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get raw data loaders
    print("Loading raw CIFAR10 data...")
    train_loader, test_loader, num_classes = get_data_loaders(config)
    print(f"Number of classes: {num_classes}")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

    # Get JPEG compressed data loaders
    print("\nLoading JPEG compressed (q=10) CIFAR10 data...")
    config['jpeg_quality'] = 10
    train_loader_jpeg, _, _ = get_data_loaders(config)
    for images, labels in train_loader_jpeg:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break