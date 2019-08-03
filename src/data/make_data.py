# data_generator.py
# Arnav Ghosh
# 3rd Aug. 2019

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_dataset(train_path, val_path):
    train_transforms =  transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation((-120, 120)),
                                            transforms.RandomAffine(0, translate=(0.5, 0.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                 std=[0.229, 0.224, 0.225])
                                            ]),
    val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomRotation((-120, 120)),
                                         transforms.RandomAffine(0, translate=(0.5, 0.5)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225])
                                        ])

    train_dataset = datasets.ImageFolder(train_path, transform = train_transforms)
    val_dataset = datasets.ImageFolder(val_path, transform = val_transforms)

    return train_dataset, val_dataset


# type of transforms to try:
# can't do random crop because that could just take a white portion
# want translation, rotation, flipping as well
# resize model


