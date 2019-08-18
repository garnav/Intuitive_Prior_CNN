# train.py
# Arnav Ghosh
# 03 Aug. 2019

#from src.data import make_data
import make_data
#from src import utils
import utils

import copy
import time
from tqdm import tqdm

import torch
import torchvision.models as models
import torch.nn as nn

######### CONSTANTS #########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### MAIN #########
def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2) #HARDCODE
    return model

def train(params, dataloaders, model, criterion, optimizer):
    t_begin = time.time()

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

            for i, (inputs, labels) in enumerate(tv_loop):
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

                #de-average to mitigate batch_num bias
                loss += l.item() * inputs.size(0)
                acc += torch.sum(preds == labels.data)
                tv_loop.set_postfix(loss = loss / ((i + 1) * params.batch_size), 
                                    accuracy = acc.item() / ((i + 1) * params.batch_size) )

        epoch_loss = loss / len(dataloaders[phase])
        epoch_acc = accuracy.double() / len(dataloaders[phase])
        epoch_time = time.time() - e_begin
        print(f'Epoch {epoch + 1}/{num_epoch} {phase} loss : {epoch_loss:.4f} {phase} accuracy : {epoch_acc:.4f} in {epoch_time // 60:.0f}:{epoch_time % 60:.0f}')

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            utils.save_checkpoint(f'{params.name}_{epoch}', True, best_model_wts, opt_dict=None, epoch_num=epoch)
        elif epoch % params.log_interval == 0:
            utils.save_checkpoint(f'{params.name}_{epoch}', False, model.state_dict(), opt_dict=None, epoch_num=epoch)

    time_elapsed = time.time() - begin
    print(f'Training complete in {time_elapsed // 60 :.0f}:{time_elapsed % 60 :.0f}')
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def main(params_path, train_path, val_path):
    params = utils.Params(params_path)

    train_loader, val_loader = make_data.load_dataset(train_path, val_path)
    dataloaders = {"train" : torch.utils.data.DataLoader(train_loader, batch_size=params.batch_size, shuffle=True), 
                   "val" : torch.utils.data.DataLoader(val_loader, batch_size=params.batch_size, shuffle=True)}

    model = get_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(params, dataloaders, model, criterion, optimizer)