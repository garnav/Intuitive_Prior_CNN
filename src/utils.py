# utils.py
# Arnav Ghosh
# 05 Aug. 2019

# Adapted from Stanford CS230-Code-Examples
# (https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py)

import json
import shutil
import torch

######### CONSTANTS #########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTENSION = "pth.tar"

######### MAIN #########
class Params(object):
    #name, image_dim, num_classes, log_interval, num_epochs, batch_size
    
    def __init__(self, param_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, params_path):
        with open(params_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        return self.__dict__
    
def save_checkpoint(pth, is_best, state_dict, opt_dict=None, epoch_num=None):
    state = { 'state_dict' : state_dict,
              'opt_dict' : opt_dict,
              'epoch' : epoch_num }

    fpath = f'{pth}.{EXTENSION}'
    torch.save(state, fpath)

    if is_best:
        shutil.copyfile(fpath,  f'best.{EXTENSION}')

def load_checkpoint(state_pth, model, optimizer=None):
    fpath = f'{state_pth}.{EXTENSION}'

    state = torch.load(fpath, device)
    if optimizer is not None:
        optimizer.load_state_dict(state['opt_dict'])
    model.load_state_dict(state['state_dict'])

    return state