# visualize.py
# Arnav Ghosh
# 22 Aug. 2019

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image

######### CONSTANTS #########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### MAIN #########
class VisDataset(Dataset):

    def __init__(self, img_paths, labels, transforms = None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        im = Image.open(self.img_paths[index])
        l = self.labels[index]

        if self.transforms is not None:
            im = self.transforms(im)

        return im, l

def visualize_results(model, img_dim, img_paths, labels):
    model.eval()
    fig = plt.figure()

    vis_transforms =  transforms.Compose([transforms.Resize(img_dim),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])
                                        ])
    vis_dataset = VisDataset(img_paths, labels, vis_transforms)
    vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for i, (inputs, label) in enumerate(vis_loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            ax = plt.subplot(len(vis_dataset) // 2, 2, i + 1)
            ax.axis('off')
            ax.set_title(f'Predicted : {preds[0]}')
            plt.imshow(inputs.cpu().data[0].permute(1,2,0))

    model.train()