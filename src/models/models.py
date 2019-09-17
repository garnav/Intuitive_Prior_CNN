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

def create_custom_model(init_dim, num_classes, pre_trained):
	return PatternNet(num_classes).to(device)

##### CUSTOM NETS #####
class PatternNet(nn.Module):
	def __init__(self, num_classes):
		super(PatternNet, self).__init__()

		self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
									  nn.ReLU(inplace=True), 
									  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
									  nn.ReLU(inplace=True), 
									  nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 
									  nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
									  nn.ReLU(inplace=True), 
									  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
									  nn.ReLU(inplace=True), 
									  nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 
									  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
									  nn.ReLU(inplace=True), 
									  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
									  nn.ReLU(inplace=True), 
									  nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
		#self.avgpool = nn.AdaptiveMaxPool2d((8, 8))
		self.classifier = nn.Sequential(nn.Linear(in_features=(64 * 64 * 64), out_features=1024, bias=True), 
										nn.ReLU(inplace=True), 
										nn.Linear(in_features=1024, out_features=256, bias=True),
										nn.ReLU(inplace=True),
										nn.Linear(in_features=256, out_features=num_classes, bias=True))

	def forward(self, x):
		out = self.features(x)
		#out = self.avgpool(out)
		out = out.reshape(out.size(0), -1)
		out = self.classifier(out)
		return out