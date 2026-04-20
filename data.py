"""CIFAR-10 data loaders with standard augmentation."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for CIFAR-10.
    Images are in [0, 1] (no mean/std normalization so that PGD constraints
    can be expressed directly in pixel space).
    """
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_tf = T.Compose([T.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
