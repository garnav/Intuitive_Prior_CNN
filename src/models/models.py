# model.py
# Arnav Ghosh
# 24 Aug. 2019

# TODO import utils
#from src import utils
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


######### CONSTANTS #########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### MAIN #########
def create_model(init_dim, num_classes, pre_trained):
    return create_resnet_model(init_dim, num_classes, pre_trained)

def load_model(init_dim, num_classes, state_path):
	model = create_model(init_dim, num_classes, False)
	utils.load_checkpoint(state_path, model, optimizer=None)
	return model

def create_resnet_model(init_dim, num_classes, pre_trained):
    model = models.resnet18(pretrained=pre_trained)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
    return model.to(device)  

def create_vgg_model(init_dim, num_classes, pre_trained):
    model = models.vgg16(pretrained=pre_trained)
    model.classifier[6] = nn.Linear(in_features = model.classifier[6].in_features, out_features = num_classes)
    return model.to(device)