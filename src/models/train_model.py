# train.py
# Arnav Ghosh
# 6 Sept. 2019

#from src.data import make_data
import make_data
#from src import utils
import utils
import models

import copy
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

######### CONSTANTS #########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### MAIN #########
def train(params, dataloaders, dataset_sizes, model, criterion, optimizer):
    t_begin = time.time()
    writer = SummaryWriter(params.log_dir)

    vis_params = list(map(lambda x : x[0] , list(model.named_parameters())))
    vis_params = list(filter(lambda x : ("bias" not in x) and ("bn" not in x) and ("downsample" not in x), vis_params))
    vis_params = [vis_params[int(len(vis_params) / 3)], 
                  vis_params[int((2 * len(vis_params)) / 3)], 
                  vis_params[len(vis_params) - 1]]

    print("Logging Gradients For: ")
    print(vis_params)

    for param in vis_params:
        model.state_dict(keep_vars=True).get(param).retain_grad()

    best_weights = copy.deepcopy(model.state_dict)
    best_acc = 0.0

    for epoch in range(params.num_epochs):
        e_begin = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss = 0.0
            acc = 0.0

            tv_loop = tqdm(dataloaders[phase])
            tv_loop.set_description(f'Epoch {epoch + 1}/{params.num_epochs} : {phase}')

            vis_i = np.random.choice(len(tv_loop)) # LOGGING

            for i, (inputs, labels) in enumerate(tv_loop):
                if vis_i == i and phase == "train":
                    hooks = []
                    vis_modules = list(map(lambda x : x[:x.rindex(".")], vis_params))
                    for name, module in model.named_modules():
                        if name in vis_modules:
                            #hooks.append(module.register_forward_hook(lambda self, input, output : write_activations(name, writer, epoch, input[0], output[0])))
                            hooks.append(module.register_forward_hook(lambda self, input, output, n=name :  write_activations(n, writer, epoch, input[0], output[0]) ))

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    pred_prob, preds = torch.max(outputs, 1)
                    l = criterion(pred_prob, labels.float())

                    if phase == 'train':
                        l.backward()
                        optimizer.step()

                        # LOGGING
                        if vis_i == i:
                            for j, param in enumerate(vis_params):
                                writer.add_histogram(f'{"Weights"}/{param}', model.state_dict(keep_vars=True).get(param), epoch)
                                writer.add_histogram(f'{"Gradients"}/{param}', model.state_dict(keep_vars=True).get(param).grad, epoch)
                                hooks[j].remove()

                #de-average to mitigate batch_num bias
                loss += l.item() * inputs.size(0)
                acc += torch.sum(preds == labels.data)
                tv_loop.set_postfix(loss = loss / ((i + 1) * params.batch_size), 
                                    accuracy = acc.item() / ((i + 1) * params.batch_size) )

            epoch_loss = loss / float(dataset_sizes[phase])
            epoch_acc = acc.item() / float(dataset_sizes[phase])
            epoch_time = time.time() - e_begin
            print(f'Epoch {epoch + 1}/{params.num_epochs} {phase} loss : {epoch_loss:.4f} \
                                                          {phase} accuracy : {epoch_acc:.4f} in {epoch_time // 60:.0f} min:{epoch_time % 60:.0f} s')

            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                utils.save_checkpoint(f'{params.weights_dir}{params.name}_{epoch}', True, best_model_wts, opt_dict=None, epoch_num=epoch)
            elif phase == 'val' and (epoch % params.log_interval == 0):
                utils.save_checkpoint(f'{params.weights_dir}{params.name}_{epoch}', False, model.state_dict(), opt_dict=None, epoch_num=epoch)
        
    time_elapsed = time.time() - t_begin
    print(f'Training complete in {time_elapsed // 60 :.0f} min : {time_elapsed % 60 :.0f} s')
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def main(params_path, train_path, val_path):
    params = utils.Params(params_path)
    
    train_loader, val_loader = make_data.load_dataset(params.image_dim, train_path, val_path)
    dataloaders = {"train" : torch.utils.data.DataLoader(train_loader, batch_size=params.batch_size, shuffle=True), 
                   "val" : torch.utils.data.DataLoader(val_loader, batch_size=params.batch_size, shuffle=True)}
                   
    dataset_sizes = {"train" : len(train_loader), "val" : len(val_loader)}

    model = models.create_model(params.image_dim, params.num_classes, True)
    
    criterion = nn.BCEWithLogitsLoss()
    #print("new opt")
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.Adam([{'params' : list(model.parameters())[:-2]}, 
                                  {'params' : model.fc.parameters(), 'lr' : 0.00001}], lr=0.001)
    #optimizer = torch.optim.SGD([{'params' : list(model.parameters())[:-2]}, 
    #                              {'params' : model.fc.parameters(), 'lr' : 0.0001}], lr=0.001)
    #optimizer = torch.optim.Adam([{'params' : model.features.parameters()}, 
    #                              {'params' : model.classifier.parameters(), 'lr' : 0.0005}], lr=0.005)

    train(params, dataloaders, dataset_sizes, model, criterion, optimizer)

def write_activations(name, writer, epoch, input, output):
    writer.add_histogram(f'{"Input"}/{name}', input, epoch)
    writer.add_histogram(f'{"Output"}/{name}', output, epoch)