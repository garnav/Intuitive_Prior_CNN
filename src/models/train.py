# train.py
# Arnav Ghosh
# 3rd Aug. 2019

import time
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    pass

def train(params, dataloaders, model, criterion, optimizer):
    t_begin = time.time()

    best_weights = copy.deepcopy(model.state_dict)
    best_acc = 0.0

    for epoch in range(params.num_epoch):
        e_begin = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss = 0.0
            acc = 0.0

            tv_loop = tqdm(dataloaders[phase])
            tv_loop.set_description(f'Epoch {epoch + 1}/{num_epoch} : {phase}')

            for inputs, labels in tv_loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    l = criterion(outputs, labels)

                    if phase == 'train':
                        l.backward()
                        optimizer.step()

                #de-average to mitigate batch_num bias
                loss += l.item() * inputs.size(0)
                acc += torch.sum(preds == labels.data)

                tv_loop.set_postfix(loss = loss.item(), accuracy = acc.item())

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

def main():
    # params
    # model
    # loss
    # optimizer
    # train
    pass