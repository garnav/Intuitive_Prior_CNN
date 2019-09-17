# data_generator.py
# Arnav Ghosh
# 18 Aug. 2019

import numpy as np
import os
from PIL import ImageOps
import shutil

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def create_training_directories(root, pos_paths, neg_paths, train_ratio):
    train_pth = os.path.join(root, "train")
    val_pth = os.path.join(root, "val")

    if (not os.path.isdir(train_pth)) and (not os.path.isdir(val_pth)):
        os.mkdir(train_pth)
        os.mkdir(val_pth)

        pos_train_pth = os.path.join(train_pth, "1")
        neg_train_pth = os.path.join(train_pth, "0")
        pos_val_pth = os.path.join(val_pth, "1")
        neg_val_pth = os.path.join(val_pth, "0")
        os.mkdir(pos_train_pth)
        os.mkdir(neg_train_pth)
        os.mkdir(pos_val_pth)
        os.mkdir(neg_val_pth)

        for path in pos_paths:
            files = os.listdir(path)
            np.random.shuffle(files)

            train_files = np.random.choice(files, int(train_ratio * len(files)), replace = False)
            val_files = list(set(files).difference(set(train_files)))

            for file in train_files:
                shutil.copyfile(os.path.join(path, file),  os.path.join(pos_train_pth, file))

            for file in val_files:
                shutil.copyfile(os.path.join(path, file),  os.path.join(pos_val_pth, file))

        for path in neg_paths:
            files = os.listdir(path)
            np.random.shuffle(files)

            train_files = np.random.choice(files, int(train_ratio * len(files)), replace = False)
            val_files = list(set(files).difference(set(train_files)))

            for file in train_files:
                shutil.copyfile(os.path.join(path, file),  os.path.join(neg_train_pth, file))

            for file in val_files:
                shutil.copyfile(os.path.join(path, file),  os.path.join(neg_val_pth, file))
    else:
        raise Exception("Training and Validation Sets already exist.")

def load_dataset(image_dim, train_path, val_path):
    train_transforms =  transforms.Compose([transforms.Resize(image_dim),
                                            ImageOps.invert,
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation((-120, 120)),
                                            transforms.RandomAffine(0, translate=(0.5, 0.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])
                                            ])
    val_transforms = transforms.Compose([transforms.Resize(image_dim),
                                         ImageOps.invert,
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomRotation((-120, 120)),
                                         transforms.RandomAffine(0, translate=(0.5, 0.5)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225])
                                        ])

    train_dataset = datasets.ImageFolder(train_path, train_transforms)
    val_dataset = datasets.ImageFolder(val_path, val_transforms)

    return train_dataset, val_dataset


# type of transforms to try:
# can't do random crop because that could just take a white portion
# want translation, rotation, flipping as well
# resize model


